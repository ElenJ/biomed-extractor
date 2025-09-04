import os
import pandas as pd
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import argparse as ag
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value


# re-assemble words into sequences (sentences)
def group_by_sentence(df):
    sentences = []
    current_sentence_words = []
    current_sentence_tags = []

    for index, row in df.iterrows():
        current_sentence_words.append(row["word"])
        current_sentence_tags.append(row["tag"])

        if row["word"] == ".":
            sentences.append({"word": current_sentence_words, "tag": current_sentence_tags})
            current_sentence_words = []
            current_sentence_tags = []

    # Add any remaining words as the last sentence if the file doesn't end with a period
    if current_sentence_words:
        sentences.append({"word": current_sentence_words, "tag": current_sentence_tags})

    return pd.DataFrame(sentences)

# Create the parser
parser = ag.ArgumentParser(description="Provide data and model")

# Add arguments
parser.add_argument("input_train", help="Train file name")
parser.add_argument("input_test", help="Test file name")
parser.add_argument("input_val", help="Validation file name")




def process_input_files_for_PICO(TRAIN_DIR, TEST_DIR, DEV_DIR):
    df_train = pd.read_csv(TRAIN_DIR, sep='\t', header=None)
    df_test = pd.read_csv(TEST_DIR, sep='\t', header=None)
    df_dev = pd.read_csv(DEV_DIR, sep='\t', header=None, quoting=3)

    # rename columns to "word", "tag" for all dataframes
    df_train.columns = ["word", "tag"]
    df_test.columns = ["word", "tag"]
    df_dev.columns = ["word", "tag"]

    print(df_train.head())
    # Apply the grouping function to your DataFrames
    df_train_grouped = group_by_sentence(df_train)
    df_test_grouped = group_by_sentence(df_test)
    df_dev_grouped = group_by_sentence(df_dev)
    dataset_train_grouped = Dataset.from_pandas(df_train_grouped)
    dataset_test_grouped = Dataset.from_pandas(df_test_grouped)
    dataset_dev_grouped = Dataset.from_pandas(df_dev_grouped)

    # Combine into a DatasetDict
    dataset_grouped = DatasetDict({
        "train": dataset_train_grouped,
        "test": dataset_test_grouped,
        "dev": dataset_dev_grouped
    })

    # Get the unique tags from the training dataframe for ClassLabel names
    unique_tags = df_train["tag"].unique().tolist()

    # Create a ClassLabel feature for your tags
    tag_feature = ClassLabel(names=unique_tags)

    # Define the features for the grouped dataset, including the ClassLabel for tags
    grouped_features = Features({
        "word": Sequence(Value(dtype='string', id=None)),
        "tag": Sequence(tag_feature, id=None) # Apply ClassLabel to the sequence of tags
    })

    # Cast the grouped dataset to the defined features to apply the ClassLabel mapping
    # Apply casting to all splits in the DatasetDict
    dataset_grouped = dataset_grouped.cast(grouped_features)

    # Display the first example in the train split of the new dataset
    print(dataset_grouped["train"][0])
    # Display the features to confirm ClassLabel is applied to the tag sequence
    print(dataset_grouped["train"].features["tag"])

    # Get the ClassLabel feature object from the dataset
    tags = dataset_grouped['train'].features["tag"].feature
    #index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    #tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    dataset_grouped = dataset_grouped.map(lambda df: create_tag_names(df, tags))

    return dataset_grouped

def create_tag_names(df, tags):
    return {"ner_tags_str": [tags.int2str(idx) for idx in df["tag"]]}




 #C:/Users/elena.jolkver/AppData/Local/miniforge3/envs/nlp/python.exe c:/Users/elena.jolkver/Documents/github/biomed_extractor/app/nlp/train_model.py ./data/pico_dataset_for_training/PICO_merged_train.txt ./data/pico_dataset_for_training/PICO_merged_test.txt ./data/pico_dataset_for_training/PICO_merged_dev.txt
if __name__ == "__main__":
    print("Hello")
    # Parse the arguments
    #args = parser.parse_args()
    #process_input_files_for_PICO(args.input_train, args.input_test, args.input_val)
    PROJECT_ROOT =  'c:\\Users\\elena.jolkver\\Documents\\github\\biomed_extractor\\data\\pico_dataset_for_training'
    # Data directory at top level
    TRAIN_DIR = os.path.join(PROJECT_ROOT, 'PICO_merged_train.txt')
    TEST_DIR = os.path.join(PROJECT_ROOT, 'PICO_merged_test.txt')
    DEV_DIR = os.path.join(PROJECT_ROOT, 'PICO_merged_dev.txt')

    input_dataset = process_input_files_for_PICO(TRAIN_DIR, TEST_DIR, DEV_DIR)

    example = input_dataset['train'][0]
    print(pd.DataFrame([example["word"], example['tag'], example["ner_tags_str"]],['Tokens', 'Tags', 'Tags_decode']))

    print("DONE")