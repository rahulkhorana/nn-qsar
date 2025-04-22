import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset

BASE_PATH = Path(__file__)
src_dir = BASE_PATH.parent.parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
data_path = src_dir.__str__() + "/data"
if str(data_path) not in sys.path:
    sys.path.insert(0, str(data_path))


def create_hf_dataset(
    csv_path: str,
    sequence_col: str = "Protein_Sequence",
    smiles_col: str = "SMILES",
    target_col: str = "pActivity",
    test_split: float = 0.2,
    seed: int = 42,
):
    """
    Create a Hugging Face Dataset from a QSAR CSV with protein sequences and SMILES.

    Args:
        csv_path (str): Path to the input CSV.
        sequence_col (str): Column name containing protein sequences.
        smiles_col (str): Column name containing SMILES.
        target_col (str): Column name containing activity values (e.g., pActivity).
        test_split (float): Fraction of data to reserve for test set.
        seed (int): Random seed for splitting.

    Returns:
        dataset (datasets.DatasetDict): Hugging Face dataset with 'train' and 'test' splits.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[sequence_col, smiles_col, target_col])
    df = df.rename(
        columns={
            sequence_col: "protein_sequence",
            smiles_col: "smiles",
            target_col: "label",
        }
    )
    df = df[["protein_sequence", "smiles", "label"]]
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=test_split, seed=seed)


dataset = create_hf_dataset(data_path + "/qsar_ic50_fasta.csv")

# print(dataset)
# print(dataset["train"][0])
