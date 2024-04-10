import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)


import pandas as pd
from grape import Graph


def main():
    lotus = Graph.from_csv(
        node_path="./data/lotus/lotus_nodes.csv",
        node_list_separator=",",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/lotus/lotus_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        # weights_column_number=3,
        edge_list_edge_types_column_number=4,
        # directed=False,
        directed=True,
    )

    species = Graph.from_csv(
        node_path="./data/species/full_wikidata_taxonomy_nodes.csv",
        node_list_separator=",",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/species/full_wikidata_taxonomy_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # weights_column_number=4,
        directed=True,
        # directed=False,
    )

    molecules_to_chemont = Graph.from_csv(
        node_path="./data/molecules/mol_to_chemont_nodes.csv",
        node_list_separator=",",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/molecules/mol_to_chemont_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # weights_column_number=4,
        directed=True,
        # directed=False,
    )

    chemont = Graph.from_csv(
        node_path="./data/molecules/chemont_nodes.csv",
        node_list_separator="\t",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=3,
        edge_path="./data/molecules/chemont_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # weights_column_number=4,
        # directed=False,
        directed=True,
    )

    molecules_to_np = Graph.from_csv(
        node_path="./data/molecules/mol_to_np_nodes.csv",
        node_list_separator="\t",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/molecules/mol_to_np_edges.csv",
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # directed=False,
        directed=True,
    )

    mol_to_mol_similarity = Graph.from_csv(
        name="mol_to_mol_similarity",
        node_path="./data/molecules/mol_to_mol_similarity_nodes.csv",
        edge_path="./data/molecules/mol_to_mol_similarity_edges.csv",
        node_list_separator=",",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=4,
        directed=True,
    )

    chemicals = chemont | molecules_to_chemont

    chemical_with_np_classifier = chemicals | molecules_to_np

    chemicals_and_lotus = chemical_with_np_classifier | lotus

    full_graph = chemicals_and_lotus | species

    full_graph.dump_nodes(
        path="./data/full_wd_taxonomy_with_molecules_in_lotus_nodes.csv",
        header=True,
        nodes_column_number=0,
        nodes_column="nodes",
        node_types_column_number=1,
        node_type_column="type",
    )

    full_graph.dump_edges(
        path="./data/full_wd_taxonomy_with_molecules_in_lotus_edges.csv",
        header=True,
        directed=True,
        edge_types_column_number=2,
        edge_type_column="type",
    )

    full_graph_clean = full_graph.remove_singleton_nodes()
    full_graph_clean = full_graph_clean.remove_components(top_k_components=1)

    # filter species with no phylogeny
    lotus_df = pd.read_csv(
        "./data/molecules/230106_frozen_metadata.csv.gz", low_memory=False
    )
    lotus_df["wd_species"] = "wd:" + lotus_df["organism_wikidata"].str.extract(
        r"(Q\d+)"
    )
    species_phylo = pd.read_csv("./data/species/full_wikidata_taxonomy_nodes.csv")
    species_to_remove = list(set(lotus_df.wd_species) - set(species_phylo.node))

    full_graph_clean = full_graph_clean.filter_from_names(
        node_names_to_remove=list(species_to_remove),
    )

    full_graph_clean.dump_nodes(
        path="./data/full_wd_taxonomy_with_molecules_in_lotus_clean_nodes.csv",
        header=True,
        nodes_column_number=0,
        nodes_column="nodes",
        node_types_column_number=1,
        node_type_column="type",
    )

    full_graph_clean.dump_edges(
        path="./data/full_wd_taxonomy_with_molecules_in_lotus_clean_edges.csv",
        header=True,
        directed=True,
        edge_types_column_number=2,
        edge_type_column="edge_type",
    )

    full_graph_clean_with_mol_to_mol_similarity = (
        full_graph_clean | mol_to_mol_similarity
    )

    full_graph_clean_with_mol_to_mol_similarity.dump_nodes(
        path="./data/full_graph_clean_nodes.csv",
        header=True,
        nodes_column_number=0,
        nodes_column="nodes",
        node_types_column_number=1,
        node_type_column="type",
    )

    full_graph_clean_with_mol_to_mol_similarity.dump_edges(
        path="./data/full_graph_clean_edges.csv",
        header=True,
        directed=True,
        edge_types_column_number=2,
        edge_type_column="edge_type",
    )


if __name__ == "__main__":
    main()
