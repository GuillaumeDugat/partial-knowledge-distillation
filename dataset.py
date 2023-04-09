import os
import requests
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from model import get_tokenizer


def download_dataset():
    subdir = "data"
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace("\\", "/")  # needed for Windows

    for ds in [
        "webtext",
    ]:
        for split in ["train", "valid", "test"]:
            filename = ds + "." + split + ".jsonl"
            r = requests.get(
                "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"
                + filename,
                stream=True,
            )

            with open(os.path.join(subdir, filename), "wb") as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(
                    ncols=100,
                    desc="Fetching " + filename,
                    total=file_size,
                    unit_scale=True,
                ) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)


def get_text_dfs(config):
    train_path = os.path.join(
        config["paths"]["data_folder"], config["paths"]["train_file"]
    )
    valid_path = os.path.join(
        config["paths"]["data_folder"], config["paths"]["valid_file"]
    )
    test_path = os.path.join(
        config["paths"]["data_folder"], config["paths"]["test_file"]
    )
    if (
        not os.path.exists(train_path)
        or not os.path.exists(valid_path)
        or not os.path.exists(test_path)
    ):
        download_dataset()

    train_df = pd.read_json(path_or_buf=train_path, lines=True)
    valid_df = pd.read_json(path_or_buf=valid_path, lines=True)
    test_df = pd.read_json(path_or_buf=test_path, lines=True)
    return train_df, valid_df, test_df


def get_tokens_dataset(config, tokenizer, max_length=768):
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

    def collate_fn(batch: list[dict[str, torch.tensor]]):
        # concatenate input ids and attention mask along batch axis
        batch: dict[str, torch.tensor] = data_collator(batch)

        # retrieve useful stuff
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]
        batch_arange = torch.arange(end=batch_size)

        # given all the ones are located at the beginning of the row, summing per row gives us
        # the index of the last 1 in the attention mask (= index of last word in sentence)
        nb_ones_per_row = attention_mask.sum(dim=1)

        # we chose at random one word the model will have to predict (between first and last word)
        # could be better if issue solved : https://github.com/pytorch/pytorch/issues/89438
        index_words_to_predict = torch.concatenate(
            [torch.randint(high=high, size=(1,)) for high in nb_ones_per_row]
        )

        # all the words situated after the word chosen at random get their attention mask set to 0
        set_to_0 = torch.tensor(
            sum(
                [
                    [
                        [j, i]
                        for i in range(index_words_to_predict[j], nb_ones_per_row[j])
                    ]
                    for j in range(batch_size)
                ],
                [],
            )
        )
        attention_mask[set_to_0[:, 0], set_to_0[:, 1]] = 0

        # we retrieve the tokens of the chosen words and create one hot encoded versions of the
        words_ids = input_ids[batch_arange, index_words_to_predict]
        y = torch.nn.functional.one_hot(words_ids, num_classes=tokenizer.vocab_size)

        x = {"input_ids": input_ids, "attention_mask": attention_mask}
        return x, y

    for i, dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
        dataloaders[i] = DataLoader(
            dataset,
            batch_size=config["training_parameters"]["batch_size"],
            collate_fn=collate_fn,
            shuffle=True,
        )
    return dataloaders


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
