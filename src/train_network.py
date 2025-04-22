import sys
from pathlib import Path

import torch
from dataset import create_hf_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from network import DualEncoderRegressor
from collator import DataCollatorForDualEncoder


BASE_PATH = Path(__file__)
src_dir = BASE_PATH.parent.parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
data_path = src_dir.__str__() + "/data"
if str(data_path) not in sys.path:
    sys.path.insert(0, str(data_path))


dataset = create_hf_dataset(data_path + "/qsar_ic50_fasta.csv")
protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
smiles_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
data_collator = DataCollatorForDualEncoder(protein_tokenizer, smiles_tokenizer)


model = DualEncoderRegressor()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)


trainer.train()
