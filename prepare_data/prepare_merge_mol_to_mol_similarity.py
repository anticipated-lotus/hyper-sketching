import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

from glob import glob

import pandas as pd
from tqdm import tqdm


def main():
    edges = (
        pd.concat(
            [
                pd.read_csv(path, index_col=0)
                for path in tqdm(
                    glob(
                        "./data/molecules/graph_of_mol_to_mol_similarity/num_files_1000/*/*/cosine_similarity_edges.csv"
                    )
                )
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    edges.to_csv("./data/molecules/mol_to_mol_similarity_edges.csv")

    nodes = pd.DataFrame(
        {
            "node": pd.concat([edges.child, edges.parent]).drop_duplicates().values,
            "type": "biolink:ChemicalEntity",
        }
    )

    nodes.to_csv("./data/molecules/mol_to_mol_similarity_nodes.csv")


if __name__ == "__main__":
    main()
