import os
import requests
from typing import List, Dict
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from model import get_tokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def get_text_dfs(config):
    dataset = load_dataset("vicgalle/alpaca-gpt4")
    dataset = pd.DataFrame(dataset["train"])  # There is only train
    X_train, X_test = train_test_split(
        dataset, test_size=0.2, shuffle=True, random_state=config["seed"]
    )
    X_train, X_val = train_test_split(
        X_train.copy(), test_size=0.25, random_state=config["seed"]
    )

    return X_train, X_val, X_test


def get_tokens_dataset(config, tokenizer, max_length=1024):
    token_dataset_path = os.path.join(
        config["paths"]["data_folder"], config["paths"]["token_dataset"]
    )
    if os.path.exists(token_dataset_path):
        with open(token_dataset_path, "r") as f:
            result = json.loads(f.read())
        for train_test_val in result.keys():
            for key, value in result[train_test_val].items():
                result[train_test_val][key] = torch.tensor(value)

    else:
        train_df, valid_df, test_df = get_text_dfs(config)
        names = ["train", "valid", "test"]
        result = {}
        for i, df in enumerate([train_df, valid_df, test_df]):
            tokens = tokenizer(
                df["text"].tolist(),
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = {**tokens}  # convert to dict
            result[names[i]] = tokens

        # convert tensors to list to save
        savable_result = {}
        for train_test_val in result.keys():
            savable_result[train_test_val] = {}
            for key, value in result[train_test_val].items():
                savable_result[train_test_val][key] = value.tolist()

        with open(token_dataset_path, "w+") as f:
            f.write(json.dumps(savable_result, indent=4))

    return (
        TokensDataset(result["train"]),
        TokensDataset(result["valid"]),
        TokensDataset(result["test"]),
    )


def get_dataloaders(config):
    tokenizer = get_tokenizer()
    train_dataset, valid_dataset, test_dataset = get_tokens_dataset(config, tokenizer)
    dataloaders = [None, None, None]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def collate_fn(batch: List[Dict[str, torch.tensor]]):
        # concatenate input ids and attention mask along batch axis
        batch: dict[str, torch.tensor] = data_collator(batch)

        # retrieve useful stuff
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]

        # we create one hot encoded versions of the input to get our target
        y = torch.nn.functional.one_hot(input_ids, num_classes=tokenizer.vocab_size)
        y = y.double()

        # remove the first word, as it's never going to be predicted
        y = y[:, 1:]

        x = {"input_ids": input_ids, "attention_mask": attention_mask}
        return x, y

    remove_tokenizer_warning()

    for i, dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
        dataloaders[i] = DataLoader(
            dataset,
            batch_size=config["training_parameters"]["batch_size"],
            collate_fn=collate_fn,
            shuffle=True,
        )
    return dataloaders


def remove_tokenizer_warning():
    # Useless warning when using tokenizer and data collator
    # https://github.com/huggingface/transformers/issues/19471
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class TokensDataset(Dataset):
    def __init__(self, token_dict) -> None:
        self.input_ids = token_dict["input_ids"]
        self.attention_mask = token_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class TextDataset(Dataset):
    def __init__(self, df) -> None:
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return self.df["text"].iloc[idx]


if __name__ == "__main__":
    import configue

    config = configue.load("config.yaml")
    train, valid, test = get_dataloaders(config)
    for x, y in train:
        print(x, y)
        break
