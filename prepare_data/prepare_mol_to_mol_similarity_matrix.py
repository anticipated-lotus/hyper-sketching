import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

import gzip

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import calculate_fingerprint_parallel


def main():
    lotus = pd.read_csv(
        "./data/molecules/230106_frozen_metadata.csv.gz", low_memory=False
    )

    lotus["wd_molecule"] = "wd:" + lotus.structure_wikidata.str.extract(r"(Q\d+)")

    wd = (
        lotus[
            [
                "wd_molecule",
                "structure_smiles_2D",
            ]
        ]
        .drop_duplicates(subset=["wd_molecule"])
        .reset_index(drop=True)
    )
    index = (
        wd.wd_molecule.str.extract(r"wd:Q(\d+)").astype("int64").sort_values(0).index
    )
    wd = wd.reindex(index).reset_index(drop=True)

    embedding_full = np.array(
        calculate_fingerprint_parallel(wd["structure_smiles_2D"].values, radi=2)
    )

    SIZE = len(embedding_full)

    embedding = embedding_full[:SIZE].astype(np.float32)

    d = embedding.shape[1]

    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embedding)
    index.add(embedding)

    # Split the similarity matrix into multiple smaller files
    num_files = 1000
    chunk_size = SIZE // num_files

    for file_num in tqdm(range(num_files)):
        start_index = file_num * chunk_size
        end_index = (file_num + 1) * chunk_size

        with gzip.open(
            f"./data/molecules/similarity_matrix/similarity_matrix_{file_num}.csv.gz",
            "w",
        ) as f:
            for i in tqdm(range(start_index, end_index)):
                D, I = index.search(embedding[i].reshape(1, -1), SIZE)
                out_array = D.reshape(-1)[np.argsort(I.reshape(-1))].astype("float16")
                out_array[:i] = 0
                np.savetxt(f, [out_array], delimiter=",")


if __name__ == "__main__":
    main()
