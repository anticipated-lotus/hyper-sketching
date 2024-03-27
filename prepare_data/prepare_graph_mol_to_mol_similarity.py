import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

import pandas as pd
import polars as pl
from cache_decorator import Cache
from tqdm import tqdm


@Cache(
    cache_dir="data/molecules/graph_of_mol_to_mol_similarity/num_files_{num_files}/{file_num}/{_hash}",
    cache_path={
        "edges": "cosine_similarity_edges.csv",
        "nodes": "cosine_similarity_nodes.csv",
    },
    use_approximated_hash=True,
)
def loop(file_num: int, num_files: int, wd: pd.DataFrame, SIZE: int, chunk_size: int):
    start = file_num * chunk_size
    end = (file_num + 1) * chunk_size
    if file_num == num_files - 1:
        end = SIZE
    wd_chunk = wd[start:end].copy()

    edges = (
        pl.read_csv(
            f"./data/molecules/similarity_matrix/similarity_matrix_{file_num}.csv.gz",
            infer_schema_length=2,
            low_memory=True,
            has_header=False,
            dtypes={
                f"column_{i+1}": pl.Float32
                for i in range(len(wd["wd_molecule"].values))
            },
        )
        .lazy()
        .with_columns(pl.Series(wd_chunk["wd_molecule"].values).alias("index"))
        .rename(
            {
                f"column_{i+1}": wd["wd_molecule"].values[i]
                for i in range(len(wd["wd_molecule"].values))
            }
        )
        .melt(
            id_vars=["index"],
            value_vars=None,
            variable_name="column",
            value_name="value",
        )
        .filter(pl.col("value") > 0.9)
        .filter(pl.col("index") != pl.col("column"))
        .rename({"index": "child", "column": "parent", "value": "weight"})
        .with_columns(pl.lit("biolink:similar_to").alias("type"))
        .collect(streaming=True)
        .to_pandas()
    )

    nodes = (
        pd.DataFrame(
            {
                "node": pd.concat([edges.child, edges.parent]),
                "type": "biolink:ChemicalEntity",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return {
        "edges": edges,
        "nodes": nodes,
    }


def main(num_files: int = 1000):
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

    SIZE = len(wd)
    chunk_size = SIZE // num_files

    for file_num in tqdm(range(num_files)):
        loop(file_num, num_files, wd, SIZE, chunk_size)


if __name__ == "__main__":
    main()
