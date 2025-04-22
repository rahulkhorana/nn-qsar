import torch
from transformers import AutoTokenizer


class DataCollatorForDualEncoder:
    def __init__(self, protein_tokenizer, smiles_tokenizer):
        self.protein_tokenizer = protein_tokenizer
        self.smiles_tokenizer = smiles_tokenizer

    def __call__(self, batch):
        protein_seqs = [item["protein_sequence"] for item in batch]
        smiles_seqs = [item["smiles"] for item in batch]
        labels = [item["label"] for item in batch]

        prot_enc = self.protein_tokenizer(
            protein_seqs, padding=True, truncation=True, return_tensors="pt"
        )
        smiles_enc = self.smiles_tokenizer(
            smiles_seqs, padding=True, truncation=True, return_tensors="pt"
        )

        return {
            "protein_input_ids": prot_enc["input_ids"],
            "protein_attention_mask": prot_enc["attention_mask"],
            "smiles_input_ids": smiles_enc["input_ids"],
            "smiles_attention_mask": smiles_enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float),
        }
