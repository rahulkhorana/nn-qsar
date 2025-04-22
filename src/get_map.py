import sys
import pickle
from pathlib import Path
from itertools import chain
from multiprocessing import Pool, cpu_count
from chembl_webresource_client.new_client import new_client


BASE_PATH = Path(__file__)
src_dir = BASE_PATH.parent.parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
data_path = src_dir.__str__() + "/data"
if str(data_path) not in sys.path:
    sys.path.insert(0, str(data_path))


targets = new_client.target.filter(target_type="SINGLE PROTEIN").only(
    ["target_chembl_id", "target_components"]
)


def extract_accessions(target):
    mapping = {}
    for comp in target.get("target_components", []):
        acc = comp.get("accession")
        if acc:
            mapping[acc] = target["target_chembl_id"]
    return mapping


if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        results = pool.map(extract_accessions, targets)
    uniprot_to_chembl = dict(chain.from_iterable(d.items() for d in results if d))
    with open("uniprot_to_chembl.pkl", "wb") as f:
        pickle.dump(uniprot_to_chembl, f)
