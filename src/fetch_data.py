import sys
import asyncio
import aiohttp
import pickle
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__)
src_dir = BASE_PATH.parent.parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
data_path = src_dir.__str__() + "/data"
if str(data_path) not in sys.path:
    sys.path.insert(0, str(data_path))

MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


uniprot_df = pd.read_csv(
    data_path + "/uniprotkb_organism_id_9606_AND_reviewed.tsv", sep="\t"
)
uniprot_ids = uniprot_df["Entry"].dropna().tolist()

print(f"num of uniprot ids is: {len(uniprot_ids)})")

activity_types = {"IC50", "Ki", "Kd"}

with open(data_path + "/uniprot_to_chembl.pkl", "rb") as f:
    uniprot_to_chembl = pickle.load(f)

print(f"chembl dict: {len(uniprot_to_chembl)}")


headers = {"Accept": "application/json"}


async def fetch_activity_and_smiles(session, target_id, uniprot_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id={target_id}&limit=1000"
    async with semaphore:
        async with session.get(url, headers=headers) as resp:
            data = await resp.json()
            results = []
            for act in data.get("activities", []):
                if act.get("standard_type") in activity_types and act.get(
                    "standard_value"
                ):
                    mol_id = act["molecule_chembl_id"]
                    mol_url = (
                        f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json"
                    )
                    try:
                        async with session.get(mol_url, headers=headers) as mol_resp:
                            if mol_resp.status != 200:
                                continue
                            mol_data = await mol_resp.json()
                            if mol_data is None:
                                continue
                            smiles = mol_data.get("molecule_structures", {}).get(
                                "canonical_smiles"
                            )
                            if smiles:
                                results.append(
                                    {
                                        "UniProt_ID": uniprot_id,
                                        "Target_ID": target_id,
                                        "Ligand_ChEMBL_ID": mol_id,
                                        "Activity_Type": act["standard_type"],
                                        "Activity_Value": act["standard_value"],
                                        "Activity_Units": act["standard_units"],
                                        "SMILES": smiles,
                                    }
                                )
                    except Exception as e:
                        print(f"[⚠️ Molecule fetch failed] {mol_id}: {e}")
            return results


async def main():
    tasks = []
    async with aiohttp.ClientSession() as session:
        for uniprot_id in uniprot_ids:
            if uniprot_id in uniprot_to_chembl:
                target_id = uniprot_to_chembl[uniprot_id]
                tasks.append(fetch_activity_and_smiles(session, target_id, uniprot_id))

        results = await asyncio.gather(*tasks)
    flat_results = [item for sublist in results if sublist for item in sublist]
    df = pd.DataFrame(flat_results)
    df.to_csv("qsar_dataset_async.csv", index=False)
    print(f"✅ Saved {len(df)} entries to qsar_dataset_async.csv")


asyncio.run(main())
