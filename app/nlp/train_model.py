import os
import re
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer
)
import torch
import argparse
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from collections import Counter, defaultdict
from seqeval.metrics import f1_score

# --------------- Data Preparation -----------------

def group_by_sentence(df):
    """
    Groups words and tags in a DataFrame into sentences, splitting on period tokens.

    Args:
        df (pd.DataFrame): DataFrame with columns ['word', 'tag'].

    Returns:
        pd.DataFrame: Each row contains ['word', 'tag'] as a list for a sentence.
    """
    sentences = []
    current_sentence_words = []
    current_sentence_tags = []
    for _, row in df.iterrows():
        current_sentence_words.append(row["word"])
        current_sentence_tags.append(row["tag"])
        if row["word"] == ".":
            sentences.append({"word": current_sentence_words, "tag": current_sentence_tags})
            current_sentence_words = []
            current_sentence_tags = []
    # Add the last sentence if file doesn't end with '.'
    if current_sentence_words:
        sentences.append({"word": current_sentence_words, "tag": current_sentence_tags})
    return pd.DataFrame(sentences)


def load_and_prepare_data(train_path, test_path, dev_path):
    """
    Loads raw (word, tag) data, groups by sentence, builds Huggingface datasets
    with ClassLabel features for tags.

    Args:
        train_path, test_path, dev_path: TSV file paths.

    Returns:
        data_dict (DatasetDict): Features:
          - 'word': List[str]
          - 'tag': List[int] (ClassLabel-mapped)
          - 'ner_tags_str': List[str]
        tags (ClassLabel): For label mapping.
    """
    # Load
    def load_df(path):
        df = pd.read_csv(path, sep='\t', header=None, quoting=3)  # quoting=3 for dev file
        df.columns = ["word", "tag"]
        return df

    train_df = load_df(train_path)
    test_df = load_df(test_path)
    dev_df = load_df(dev_path)
    # Sentence grouping
    grouped = {
        "train": group_by_sentence(train_df),
        "test": group_by_sentence(test_df),
        "dev": group_by_sentence(dev_df)
    }
    # Huggingface datasets
    datasets_by_split = {
        split: Dataset.from_pandas(grouped_df)
        for split, grouped_df in grouped.items()
    }
    # Use train's unique tags for consistent indexing
    unique_tags = train_df["tag"].unique().tolist()
    tag_feature = ClassLabel(names=unique_tags)
    grouped_features = Features({
        "word": Sequence(Value("string")),
        "tag": Sequence(tag_feature)
    })
    data_dict = DatasetDict({
        split: dset.cast(grouped_features)
        for split, dset in datasets_by_split.items()
    })
    # Add decoded tag strings for convenience
    tag_vocab = data_dict['train'].features["tag"].feature
    data_dict = data_dict.map(
        lambda example: {"ner_tags_str": [tag_vocab.int2str(idx) for idx in example["tag"]]}
    )
    return data_dict, tag_vocab


def check_label_distribution(dataset):
    """
    Prints and returns B-tag entity distributions across splits.
    """
    split2freqs = defaultdict(Counter)
    for split, dset in dataset.items():
        for row in dset["ner_tags_str"]:
            for tag in row:
                if tag.startswith("B-"):
                    tag_type = tag.split("-")[1]
                    split2freqs[split][tag_type] += 1
    overview = pd.DataFrame.from_dict(split2freqs, orient="index")
    print("Check your label distribution:")
    print(overview)
    return overview

# --------------- Model Prep -----------------

def load_tokenizer(model_name):
    """
    Loads Huggingface tokenizer by name.
    """
    return AutoTokenizer.from_pretrained(model_name)

def index_tag_mappings(tag_vocab):
    """
    Provides int<->str mappings for tag vocabulary.

    Args:
        tag_vocab (ClassLabel)
    Returns:
        index2tag (dict), tag2index (dict)
    """
    index2tag = {idx: tag for idx, tag in enumerate(tag_vocab.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tag_vocab.names)}
    return index2tag, tag2index

def config_model(model_name, tag_vocab):
    """
    Loads the config and attaches tag mappings.
    """
    index2tag, tag2index = index_tag_mappings(tag_vocab)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=tag_vocab.num_classes,
        id2label=index2tag,
        label2id=tag2index
    )
    return config, index2tag, tag2index

def load_model(model_name, model_config):
    """
    Loads and returns Huggingface token classification model on right device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=model_config)
    return model.to(device)

# --------------- Tokenization & Alignment --------------

def tokenize_and_align_labels(examples, tag2index, index2tag, tokenizer):
    """
    Tokenizes sentences and aligns NER tag ids to tokens,
    Uses -100 as ignore index for subword tokens (except 'I-' labels).
    Includes a filter to skip None tokens.
    
    Args:
        examples: Dict with lists for 'word', 'tag'
        tag2index: str->int mapping
        index2tag: int->str mapping
        tokenizer: model tokenizer

    Returns:
        tokenized inputs dict
    """
    # Join only non-None tokens to sentence strings
    sentence_strs = [
        " ".join([word for word in words if word is not None]) 
        for words in examples["word"]
    ]
    
    tokenized_inputs = tokenizer(
        sentence_strs,
        truncation=True,
        padding='max_length',
        max_length=512
    )

    all_labels = []
    for i, words in enumerate(examples["word"]):
        example_tags = examples["tag"][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        aligned_labels = [-100] * len(word_ids)
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(example_tags):
                continue
            if previous_word_idx != word_idx:
                aligned_labels[token_idx] = example_tags[word_idx]
            else:
                tag_id = example_tags[word_idx]
                tag_name = index2tag.get(tag_id)
                if tag_name and tag_name.startswith("I-"):
                    aligned_labels[token_idx] = tag_id
            previous_word_idx = word_idx
        all_labels.append(aligned_labels)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

def tokenize_dataset(dataset, tag_vocab, tokenizer):
    """
    Applies batching tokenization and alignment to all splits, returns new DatasetDict.
    """
    index2tag, tag2index = index_tag_mappings(tag_vocab)
    features = Features({
        "input_ids": Sequence(Value("int64"), length=512),
        "token_type_ids": Sequence(Value("int64"), length=512),
        "attention_mask": Sequence(Value("int64"), length=512),
        "labels": Sequence(Value("int64"), length=512),
    })
    return dataset.map(
        tokenize_and_align_labels,
        batched=True,
        features=features,
        fn_kwargs={
            "tag2index": tag2index,
            "index2tag": index2tag,
            "tokenizer": tokenizer
        },
        remove_columns=["word", "tag", "ner_tags_str"]
    )

# --------------- Training -------------------

def make_model_output_dir(model_name, script_path):
    """
    Generates output directory for fine-tuned model under app/model/ relative to script.
    """
    base_dir = os.path.dirname(os.path.abspath(script_path))
    model_dir = os.path.join(base_dir, '..', 'model')
    model_dir = os.path.abspath(model_dir)
    safe_modelname = re.sub(r'[^\w\-_.]', '_', model_name)
    model_output_dir = os.path.join(model_dir, safe_modelname + '_PICO')
    os.makedirs(model_output_dir, exist_ok=True)
    return model_output_dir

def make_training_args(tokenized_dataset, num_epochs, batch_size, model_output_dir):
    """
    Constructs TrainingArguments object for Huggingface Trainer.
    """
    logging_steps = max(1, len(tokenized_dataset["train"]) // batch_size)
    return TrainingArguments(
        output_dir=model_output_dir,
        log_level="error",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_steps=int(1e6),
        weight_decay=0.01,
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        report_to="none"
    )

# Evaluation metrics: must know tag mapping!
def make_compute_metrics(index2tag):
    """
    Returns a metric computation function that uses the given index2tag mapping for seqeval.

    Args:
        index2tag: Dictionary int->str mapping
    Returns:
        compute_metrics: Callable for Trainer
    """
    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []
        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(index2tag[preds[batch_idx][seq_idx]])
            labels_list.append(example_labels)
            preds_list.append(example_preds)
        return preds_list, labels_list

    def compute_metrics(eval_pred):
        y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
        return {"f1": f1_score(y_true, y_pred)}

    return compute_metrics


def train_model(
    model_name,
    dataset_dict,
    tag_vocab,
    num_epochs,
    batch_size,
    script_path
):
    """
    Runs end-to-end tokenization, model preparation, and Huggingface Trainer training.

    Args:
        model_name: Huggingface model repo or path.
        dataset_dict: Sentence-level grouped DatasetDict.
        tag_vocab: Huggingface ClassLabel for tags.
        num_epochs, batch_size: Training hyperparameters.
        script_path: __file__ or equivalent for output dir logic.
    """
    tokenizer = load_tokenizer(model_name)
    config, index2tag, tag2index = config_model(model_name, tag_vocab)
    model = load_model(model_name, config)
    tokenized_dataset = tokenize_dataset(dataset_dict, tag_vocab, tokenizer)

    print("Tokenized sample shapes:",
          torch.tensor(tokenized_dataset["train"][0]['input_ids']).shape,
          torch.tensor(tokenized_dataset["train"][0]['labels']).shape)

    model_output_dir = make_model_output_dir(model_name, script_path)
    print("Fine-tuned model output will go to:", model_output_dir)

    training_args = make_training_args(tokenized_dataset, num_epochs, batch_size, model_output_dir)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = make_compute_metrics(index2tag)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"]
    )

    # Uncomment to train
    trainer.train()
    print("Trainer ready.")

# --------------- Main Script ---------------------

def main(args=None):
    """
    Entrypoint for model pipeline. Can be called from CLI or imported.
    """
    parser = argparse.ArgumentParser(description="Train/fine-tune a PICO NER model on token classification task.")
    parser.add_argument("--train", type=str, help="Path to train .txt file", required=False)
    parser.add_argument("--test", type=str, help="Path to test .txt file", required=False)
    parser.add_argument("--dev", type=str, help="Path to dev .txt file", required=False)
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--model-name", type=str, default="nlpie/bio-mobilebert", help="Huggingface base model")
    parser.add_argument("--use-hardcoded-paths", action="store_true",
                        help="Use fixed dataset paths (for notebook testing)")
    parser.add_argument("--script-path", type=str, default=__file__, help="Used for relative output directory (usually __file__)")

    if args is not None:
        parsed_args = parser.parse_args(args)
    else:
        parsed_args = parser.parse_args()

    if parsed_args.use_hardcoded_paths:
        PROJECT_ROOT = os.path.expanduser('~/Documents/github/biomed_extractor')
        parsed_args.train = os.path.join(PROJECT_ROOT, 'PICO_merged_train.txt')
        parsed_args.test = os.path.join(PROJECT_ROOT, 'PICO_merged_test.txt')
        parsed_args.dev = os.path.join(PROJECT_ROOT, 'PICO_merged_dev.txt')

    print("Loading and preprocessing data...")
    dataset_dict, tag_vocab = load_and_prepare_data(parsed_args.train, parsed_args.test, parsed_args.dev)
    example = dataset_dict['train'][0]
    print("Tokens/Tags for a training sentence:\n",
          pd.DataFrame([example["word"], example['tag'], example["ner_tags_str"]],
                       ['Tokens', 'Tags', 'Tags_decode'])
    )

    check_label_distribution(dataset_dict)

    train_model(
        model_name=parsed_args.model_name,
        dataset_dict=dataset_dict,
        tag_vocab=tag_vocab,
        num_epochs=parsed_args.epochs,
        batch_size=parsed_args.batch_size,
        script_path=parsed_args.script_path
    )

def colab_main(
    train_file,
    test_file,
    dev_file,
    epochs=3,
    batch_size=8,
    model_name="nlpie/bio-mobilebert",
    model_output_dir="colab_output_model"
):
    """
    Colab-friendly entry point, for usage on Colab and google's GPU
    This function calls the same internal functions and follows the same structure as main(),
    but all parameters are passed explicitly (not via argparse or CLI).
    
    Args:
        train_file, test_file, dev_file: Paths to data files
        epochs, batch_size: Training hyperparameters
        model_name: Huggingface model
        model_output_dir: Where to save the model (relative or absolute path)
    """
    print("Loading and preprocessing data...")
    dataset_dict, tag_vocab = load_and_prepare_data(train_file, test_file, dev_file)
    example = dataset_dict['train'][0]
    print("Tokens/Tags for a training sentence:\n",
          pd.DataFrame([example["word"], example['tag'], example["ner_tags_str"]],
                       ['Tokens', 'Tags', 'Tags_decode'])
    )
    check_label_distribution(dataset_dict)
    
    # For Colab, script_path has no meaning, so just use model_output_dir directly.
    train_model(
        model_name=model_name,
        dataset_dict=dataset_dict,
        tag_vocab=tag_vocab,
        num_epochs=epochs,
        batch_size=batch_size,
        script_path=model_output_dir  # Will resolve output dir inside train_model
    )

# Example of use in a Colab cell:
# colab_main('PICO_merged_train.txt', 'PICO_merged_test.txt', 'PICO_merged_dev.txt', epochs=3, batch_size=8, model_name="nlpie/compact-biobert", model_output_dir="output_model")

if __name__ == "__main__":
    main()