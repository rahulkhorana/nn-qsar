import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoConfig, AutoModelForSequenceClassification


class DualEncoderRegressor(nn.Module):
    def __init__(
        self,
        protein_model_name="facebook/esm2_t6_8M_UR50D",
        smiles_model_name="seyonec/ChemBERTa-zinc-base-v1",
        hidden_size=768,
        dropout=0.1,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            protein_model_name, num_labels=1, problem_type="regression"
        )
        self.protein_encoder = AutoModelForSequenceClassification.from_pretrained(
            protein_model_name, config=config
        )
        self.smiles_encoder = AutoModel.from_pretrained(smiles_model_name)
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        for param in self.smiles_encoder.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1088, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(
        self,
        protein_input_ids,
        protein_attention_mask,
        smiles_input_ids,
        smiles_attention_mask,
        **kwargs,
    ):
        prot_out = self.protein_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1][:, 0]
        smiles_out = self.smiles_encoder(
            input_ids=smiles_input_ids, attention_mask=smiles_attention_mask
        ).last_hidden_state[:, 0]
        x = torch.cat((prot_out, smiles_out), dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x
