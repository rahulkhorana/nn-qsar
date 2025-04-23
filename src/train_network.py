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
coallator = DataCollatorForDualEncoder(protein_tokenizer, smiles_tokenizer)


model = DualEncoderRegressor()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-6,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    warmup_steps=500,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rmse",
    greater_is_better=False,
    remove_unused_columns=False,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=coallator,
)

trainer.train()
trainer.log_metrics("train", trainer.state.global_step, trainer.state.log_history)
trainer.save_model("./results")
trainer.evaluate()
