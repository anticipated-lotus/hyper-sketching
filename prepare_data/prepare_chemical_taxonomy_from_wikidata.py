import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

import pandas as pd

from src.wikidata import (
    get_class_or_subclass_of_chemical_structures,
    get_wikidata_chemical_class_taxonomy,
)


def main():
    chemical_taxonomy = get_wikidata_chemical_class_taxonomy()

    chemical_taxonomy.rename(
        columns={
            "class": "child",
        },
        inplace=True,
    )

    chemical_taxonomy.drop(columns=["instance"], inplace=True)
    chemical_taxonomy.dropna().drop_duplicates().reset_index(drop=True, inplace=True)

    # Add "wd:" prefix to the child and parent columns
    chemical_taxonomy["child"] = "wd:" + chemical_taxonomy["child"].str.extract(
        r"(Q\d+)"
    )
    chemical_taxonomy["parent"] = "wd:" + chemical_taxonomy["parent"].str.extract(
        r"(Q\d+)"
    )

    # Add a edge type column with value "biolink:subclass_of"
    chemical_taxonomy["type"] = "biolink:subclass_of"

    # drop NaN one more time
    chemical_taxonomy.dropna(inplace=True)
    chemical_taxonomy.reset_index(drop=True, inplace=True)

    # Save the full_taxonomy dataframe to a CSV file
    chemical_taxonomy.to_csv("./data/molecules/wikidata_chemical_taxonomy_edges.csv")

    # Create a dataframe with unique node values and "biolink:OrganismTaxon" type
    nodes = pd.DataFrame(
        {
            "node": pd.concat([chemical_taxonomy.child, chemical_taxonomy.parent])
            .drop_duplicates()
            .values,
            "type": "biolink:ChemicalEntity",
        }
    )

    # Save the species_nodes dataframe to a CSV file
    nodes.to_csv("./data/molecules/wikidata_chemical_taxonomy_nodes.csv")


def link_strucutre_to_chemical_taxonommy():
    edges = get_class_or_subclass_of_chemical_structures()

    edges.rename(
        columns={
            "class": "parent",
            "structure": "child",
        },
        inplace=True,
    )

    edges.drop(columns=["instance"], inplace=True)
    edges.dropna().drop_duplicates().reset_index(drop=True, inplace=True)

    # Add "wd:" prefix to the child and parent columns
    edges["child"] = "wd:" + edges["child"].str.extract(r"(Q\d+)")
    edges["parent"] = "wd:" + edges["parent"].str.extract(r"(Q\d+)")

    # Add a edge type column with value "biolink:subclass_of"
    edges["type"] = "biolink:subclass_of"

    # drop NaN one more time
    edges.dropna(inplace=True)
    edges.reset_index(drop=True, inplace=True)

    # Save the full_taxonomy dataframe to a CSV file
    edges.to_csv("./data/molecules/wikidata_chemical_structure_to_subclass_edges.csv")

    # Create a dataframe with unique node values and "biolink:OrganismTaxon" type
    nodes = pd.DataFrame(
        {
            "node": pd.concat([edges.child, edges.parent]).drop_duplicates().values,
            "type": "biolink:ChemicalEntity",
        }
    )

    # Save the species_nodes dataframe to a CSV file
    nodes.to_csv("./data/molecules/wikidata_chemical_structure_to_subclass_nodes.csv")


if __name__ == "__main__":
    main()
    link_strucutre_to_chemical_taxonommy()
