from ragas_experimental import Dataset, experiment



def load_dataset():
    # Create a dataset
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )