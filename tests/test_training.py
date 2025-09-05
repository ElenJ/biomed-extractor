import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from datasets import DatasetDict, Dataset, ClassLabel
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Assuming the functions are imported from your main script module,
# e.g., from train_model import group_by_sentence, load_and_prepare_data, ...

from app.nlp.train_model import (
    group_by_sentence, load_and_prepare_data, check_label_distribution,
    index_tag_mappings, config_model, tokenize_and_align_labels, 
    load_tokenizer
)


def test_group_by_sentence():
    # Create a sample dataframe
    df = pd.DataFrame({
        "word": ["The", "cat", ".", "A", "dog", "."],
        "tag": ["O", "B-ANIMAL", "O", "O", "B-ANIMAL", "O"]
    })
    grouped = group_by_sentence(df)
    assert isinstance(grouped, pd.DataFrame)
    # Should be 2 sentences
    assert len(grouped) == 2
    # Each row is a list of words
    assert grouped["word"].apply(type).eq(list).all()


def test_load_and_prepare_data(tmp_path):
    # Create mock train/test/dev files
    content = "The\tO\ncat\tB-ANIMAL\n.\tO\nA\tO\ndog\tB-ANIMAL\n.\tO\n"
    fpaths = []
    for split in ["train", "test", "dev"]:
        f = tmp_path / f"{split}.txt"
        f.write_text(content)
        fpaths.append(str(f))
    dataset_dict, tag_vocab = load_and_prepare_data(*fpaths)
    assert isinstance(dataset_dict, DatasetDict)
    assert "train" in dataset_dict
    # Check content
    train = dataset_dict["train"]
    ex = train[0]
    assert isinstance(ex["word"], list)
    assert isinstance(ex["tag"], list)
    assert "ner_tags_str" in ex
    assert set(tag_vocab.names) == {"O", "B-ANIMAL"}


def test_check_label_distribution():
    # Fake dataset
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "word": [["A", "dog", "."]],
            "tag": [[0, 1, 0]],  # 0="O", 1="B-ANIMAL"
            "ner_tags_str": [["O", "B-ANIMAL", "O"]]
        }),
        "dev": Dataset.from_dict({
            "word": [["The", "cat", "."]],
            "tag": [[0, 1, 0]],
            "ner_tags_str": [["O", "B-ANIMAL", "O"]]
        }),
        "test": Dataset.from_dict({
            "word": [["A", "cat", "."]],
            "tag": [[0, 1, 0]],
            "ner_tags_str": [["O", "B-ANIMAL", "O"]]
        }),
    })
    # Should count B-ANIMAL per split
    df = check_label_distribution(dataset)
    assert "ANIMAL" in df.columns
    assert (df.fillna(0) >= 0).all().all()


def test_index_tag_mappings():
    mytags = ["O", "B-ANIMAL", "I-ANIMAL"]
    tag_vocab = ClassLabel(names=mytags)
    index2tag, tag2index = index_tag_mappings(tag_vocab)
    assert tag2index["O"] == 0
    assert index2tag[2] == "I-ANIMAL"


def test_tokenize_and_align_labels_basic():
    # "The dog ."   tags: O, B-ANIMAL, O
    examples = {
        "word": [["The", "dog", "."]],
        "tag": [[0, 1, 0]],  # O, B-ANIMAL, O
    }
    mytags = ["O", "B-ANIMAL"]
    tag_vocab = ClassLabel(names=mytags)
    index2tag, tag2index = index_tag_mappings(tag_vocab)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    output = tokenize_and_align_labels(examples, tag2index, index2tag, tokenizer)
    assert "input_ids" in output
    assert "labels" in output
    # Should preserve length match
    assert len(output["input_ids"][0]) == len(output["labels"][0])


def test_tokenize_and_align_labels_filter_none():
    # Test Nones don't crash
    examples = {
        "word": [["The", None, "dog", "."]],
        "tag": [[0, 0, 1, 0]],
    }
    mytags = ["O", "B-ANIMAL"]
    tag_vocab = ClassLabel(names=mytags)
    index2tag, tag2index = index_tag_mappings(tag_vocab)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    output = tokenize_and_align_labels(examples, tag2index, index2tag, tokenizer)
    assert "input_ids" in output
    assert "labels" in output


def test_config_model():
    mytags = ["O", "B-ANIMAL", "I-ANIMAL"]
    tag_vocab = ClassLabel(names=mytags)
    config, index2tag, tag2index = config_model("bert-base-uncased", tag_vocab)
    assert hasattr(config, "num_labels")
    assert isinstance(index2tag, dict)
    assert isinstance(tag2index, dict)


@pytest.mark.skip("Only run with internet/model access")
def test_load_tokenizer_model():
    # This test will actually download model!
    tokenizer = load_tokenizer("bert-base-uncased")
    assert tokenizer.pad_token_id is not None
    from transformers import AutoConfig
    from app.nlp.train_model import load_model
    tag_vocab = ClassLabel(names=["O", "B-ANIMAL"])
    config, index2tag, tag2index = config_model("bert-base-uncased", tag_vocab)
    model = load_model("bert-base-uncased", config)
    assert hasattr(model, "forward")


if __name__ == "__main__":
    import sys
    # To run these from the command line:
    # python test_biomed_extractor.py
    pytest.main(sys.argv)