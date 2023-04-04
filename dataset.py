import os
import requests
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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


def get_dataset(config):
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
    else:
        train_df = pd.read_json(path_or_buf=train_path, lines=True)
        train_dataset = TextDataset(train_df)
        valid_df = pd.read_json(path_or_buf=valid_path, lines=True)
        valid_dataset = TextDataset(valid_df)
        test_df = pd.read_json(path_or_buf=test_path, lines=True)
        test_dataset = TextDataset(test_df)
        return train_dataset, valid_dataset, test_dataset


def get_dataloaders(config):
    train_dataset, valid_dataset, test_dataset = get_dataset(config)
    return (
        DataLoader(train_dataset),
        DataLoader(valid_dataset),
        DataLoader(test_dataset),
    )


class TextDataset(Dataset):
    def __init__(self, df, text_column="text") -> None:
        self.df = df
        self.text_column = text_column

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return self.df.loc[idx][self.text_column]


if __name__ == "__main__":
    import configue

    config = configue.load("config.yaml")
    train, valid, test = get_dataloaders(config)
    for list_text in train:
        print(list_text)
        break
