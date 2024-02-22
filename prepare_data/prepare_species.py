import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

from src.wikidata import taxonomy_in_edges, get_full_taxonomy_of_wikidata
from tqdm import tqdm
import pandas as pd

tqdm.pandas()


def main():
    full_taxonomy = get_full_taxonomy_of_wikidata()

    full_taxonomy.rename(
        columns={
            "taxon.value": "child",
            "parent.value": "parent",
        },
        inplace=True,
    )
    full_taxonomy.dropna().drop_duplicates().reset_index(drop=True, inplace=True)
    full_taxonomy.drop(columns=["taxon.type", "parent.type"], inplace=True)

    # Add "wd:" prefix to the child and parent columns
    full_taxonomy["child"] = "wd:" + full_taxonomy["child"].str.extract(r"(Q\d+)")
    full_taxonomy["parent"] = "wd:" + full_taxonomy["parent"].str.extract(r"(Q\d+)")

    # Add a edge type column with value "biolink:subclass_of"
    full_taxonomy["type"] = "biolink:subclass_of"

    # drop NaN one more time
    full_taxonomy.dropna(inplace=True)
    full_taxonomy.reset_index(drop=True, inplace=True)

    # Save the full_taxonomy dataframe to a CSV file
    full_taxonomy.to_csv("./data/species/full_wikidata_taxonomy_edges.csv")

    # Create a dataframe with unique node values and "biolink:OrganismTaxon" type
    species_nodes = pd.DataFrame(
        {
            "node": pd.concat([full_taxonomy.child, full_taxonomy.parent])
            .drop_duplicates()
            .values,
            "type": "biolink:OrganismTaxon",
        }
    )

    # Save the species_nodes dataframe to a CSV file
    species_nodes.to_csv("./data/species/full_wikidata_taxonomy_nodes.csv")


if __name__ == "__main__":
    main()
