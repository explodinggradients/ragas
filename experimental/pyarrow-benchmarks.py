import pickle
import time

import numpy as np
import psutil
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader


def calculate_allocated_memory():
    arr = np.ones((1024, 1024, 1024, 3), dtype=np.uint8)
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print("Allocated memory: {:0.2f} MB".format(mem))


def load_hf_dataset() -> Dataset:
    # the dataset is only 18mb but the program take 540mb
    ds = load_dataset("BeIR/fiqa", "corpus")
    assert isinstance(ds, DatasetDict)
    return ds["corpus"]


def new_hf_dataset(ds):
    data = {
        "ids": ds["corpus"]["_id"],
        "text": ds["corpus"]["text"],
        "title": ds["corpus"]["title"],
    }
    # save so we can load it later
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)
    ds = Dataset.from_dict(data)


def load_hf_dataset_from_pickle():
    # seems like this is not as expensive as downloading from hf dataset hub
    # total memory: 473mb
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    ds = Dataset.from_dict(data)


def blowup_mem(ds):
    # this does take more space because pyarrow dataset is converted into
    # py_list
    for i in range(5):
        l = ds["corpus"]["text"]


def load_with_torch_from_dataset(ds):
    ds = ds.with_format("torch")
    dataloader = DataLoader(ds, batch_size=16, num_workers=0)
    for batch in dataloader:
        len(batch)


def load_with_torch_from_list(l: list[str]):
    dataloader = DataLoader(l, batch_size=16, num_workers=0)
    for batch in dataloader:
        len(batch)


def add_str(row):
    for r in row["text"]:
        r += "a"
    return row


def map_vs_forloop(ds: Dataset):
    start = time.perf_counter()
    ds.map(add_str, batched=True, batch_size=50)
    print(f"map: {time.perf_counter() - start}")

    start = time.perf_counter()
    for r in ds["text"]:
        r += "a"
    print(f"for loop: {time.perf_counter() - start}")


if __name__ == "__main__":
    """
    to-test
    1. test: https://pythonspeed.com/articles/measuring-memory-python/
    2. see if hf_dataset -> list -> hf_dataset is expensive: it is very expensive, I think this is creating
    a new dataset in memory
    3. but if we load from pickle, it is less expensive, we have a 200mb diff
    4. how is using hf_dataset and torch dataload: the diff between directly loading and load from list is not as bad as I expected.
    5. tokenize the batches in dataloader

    6. map vs for loop
    """
    ds = load_hf_dataset()
    # new_hf_dataset(ds)
    # load_hf_dataset_from_pickle()
    # load_with_torch_from_list(ds["text"])
    # load_with_torch_from_dataset(ds)
    map_vs_forloop(ds)
