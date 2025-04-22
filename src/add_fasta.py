import sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__)
src_dir = BASE_PATH.parent.parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
data_path = src_dir.__str__() + "/data"
if str(data_path) not in sys.path:
    sys.path.insert(0, str(data_path))


df = pd.read_csv(data_path + "/qsar_ic50.csv")
unique_ids = df["UniProt_ID"].unique()


def fetch_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return "".join(r.text.split("\n")[1:])
        else:
            return None
    except Exception as e:
        print(f"Error fetching {uniprot_id}: {e}")
        return None


uniprot_to_seq = {uid: fetch_sequence(uid) for uid in unique_ids}
df["Protein_Sequence"] = df["UniProt_ID"].map(uniprot_to_seq)
df = df.dropna(subset=["Protein_Sequence"])
df.reset_index(drop=True, inplace=True)
df.to_csv(data_path + "/qsar_ic50_fasta.csv", index=False)
print(f"Added sequences. Final shape: {df.shape}")
