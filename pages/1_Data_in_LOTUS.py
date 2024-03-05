import streamlit as st
import polars as pl
from src.models import XGBoost
from grape import Graph
from ensmallen import HyperSketchingPy
import pandas as pd
import numpy as np


# apply a function that looks at the species, anf if the molecule id is in the neighborhood of the species,
# then it should write in a new column called "note" this : "The link between this molecules and this species is already in LOTUS."
def check_if_in_lotus(species, molecule, graph: Graph):
    if molecule in graph.get_neighbour_node_names_from_node_name(species):
        return "This link is already in LOTUS."
    return np.nan


# Load the data
if "model" not in st.session_state:
    st.session_state["model"] = XGBoost.load_model("lightgbm_model.pkl")

if "graph" not in st.session_state:
    st.session_state["graph"] = Graph.from_csv(
        # name="lotus_with_ncbi",
        # node_path="./data/full_graph_with_ncbi_clean_nodes.csv",
        # edge_path="./data/full_graph_with_ncbi_clean_edges.csv",
        name="lotus_with_wikidata",
        node_path="./data/full_wd_taxonomy_with_molecules_in_lotus_clean_nodes.csv",
        edge_path="./data/full_wd_taxonomy_with_molecules_in_lotus_clean_edges.csv",
        node_list_separator="\t",
        node_list_header=True,
        nodes_column_number=0,
        node_list_node_types_column_number=1,
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column_number=0,
        destinations_column_number=1,
        edge_list_edge_types_column_number=2,
        # directed=True,
        directed=False,
        load_edge_list_in_parallel=False,
        load_node_list_in_parallel=False,
    )

if "sketching_features" not in st.session_state:
    st.session_state["sketching_features"] = HyperSketchingPy(
        hops=2,
        normalize=False,
        graph=st.session_state["graph"],
    )
    st.session_state["sketching_features"].fit()


if "lotus" not in st.session_state:
    st.session_state["lotus"] = pl.read_csv(
        "data/molecules/230106_frozen_metadata.csv.gz",
        dtypes={
            "structure_xlogp": pl.Float32,
            "structure_cid": pl.UInt32,
            "organism_taxonomy_ncbiid": pl.UInt32,
            "organism_taxonomy_ottid": pl.UInt32,
            "structure_stereocenters_total": pl.UInt32,
            "structure_stereocenters_unspecified": pl.UInt32,
        },
        infer_schema_length=50000,
        null_values=["", "NA"],
    )
    st.session_state["lotus"] = st.session_state["lotus"].with_columns(
        pl.col("organism_taxonomy_gbifid")
        .map_elements(lambda x: np.nan if x.startswith("c(") else x)
        .cast(pl.UInt32)
        .alias("organism_taxonomy_gbifid")
    )


if "species_phylo" not in st.session_state:
    st.session_state["species_phylo"] = pd.read_csv(
        "./data/species/full_wikidata_taxonomy_nodes.csv"
    )

# Page setup
st.set_page_config(page_title="Anticipating LOTUS", page_icon="🪷", layout="wide")
st.title("Anticipating LOTUS")


lotus = st.session_state["lotus"]
options = ["Molecule", "Species"]
selected_option = st.selectbox(
    """Select if you want to predict the occurence of a molecule in all species (Molecule)
    or if you want to find the most probable molecules present in a species (Species).
    """,
    options,
)
text_search = st.text_input(
    """Search for the molecule or species of interest.
    Molecules can be searched by their Wikidata ID, IUPAC name, SMILES, InChIKey, CID, or exact mass.
    Species can be searched by their name, Wikidata ID, NCBI ID, OTT ID, or GBIF ID.
    """,
    value="",
)


if selected_option == "Molecule":
    mask = lotus.filter(
        (pl.col("structure_wikidata").str.contains("(?i)" + text_search))
        | (pl.col("structure_smiles_2D").str.contains("(?i)" + text_search))
        | (pl.col("structure_inchikey").str.contains("(?i)" + text_search))
        | (pl.col("structure_exact_mass").str.contains(text_search, literal=True))
        | (pl.col("structure_cid").str.contains(text_search, literal=True))
        | (pl.col("structure_nameIupac").str.contains("(?i)" + text_search))
    )

    mask = mask.with_columns(
        ("wd:" + pl.col("structure_wikidata").str.extract(r"(Q\d+)")).alias(
            "wd_molecule"
        )
    )

    mask_out = mask.select(
        [
            "wd_molecule",
            "structure_wikidata",
            "structure_smiles_2D",
            "structure_inchikey",
            "structure_cid",
            "structure_exact_mass",
        ]
    )

    mask_out = mask_out.unique()

if selected_option == "Species":
    mask = lotus.filter(
        (pl.col("organism_wikidata").str.contains("(?i)" + text_search))
        | (pl.col("organism_name").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_gbifid").str.contains(text_search, literal=True))
        | (pl.col("organism_taxonomy_ncbiid").str.contains(text_search, literal=True))
        | (pl.col("organism_taxonomy_ottid").str.contains(text_search, literal=True))
        | (pl.col("organism_taxonomy_01domain").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_02kingdom").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_03phylum").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_04class").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_05order").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_06family").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_08genus").str.contains("(?i)" + text_search))
        | (pl.col("organism_taxonomy_09species").str.contains("(?i)" + text_search))
    )

    mask = mask.with_columns(
        ("wd:" + pl.col("organism_wikidata").str.extract(r"(Q\d+)")).alias("wd_species")
    )

    mask_out = mask.select(
        [
            "wd_species",
            "organism_wikidata",
            "organism_name",
            "organism_taxonomy_gbifid",
            "organism_taxonomy_ncbiid",
            "organism_taxonomy_ottid",
            "organism_taxonomy_01domain",
            "organism_taxonomy_02kingdom",
            "organism_taxonomy_03phylum",
            "organism_taxonomy_04class",
            "organism_taxonomy_05order",
            "organism_taxonomy_06family",
            "organism_taxonomy_08genus",
            "organism_taxonomy_09species",
        ]
    )
    mask_out = mask_out.unique()

# Show the results, if you have a text_search
if text_search:
    st.write(mask_out.to_pandas())


prediction = st.text_input(
    """Now you can copy the value in the column `wd_molecule` or `wd_species` and paste in the box below to get the predictions.""",
    value="",
)

threshold = st.slider(
    "Select the threshold for the predictions",
    min_value=0.0,
    max_value=1.0,
    value=(0.75, 1.0),
    step=0.01,
)

if prediction and selected_option == "Molecule":

    lotus = lotus.with_columns(
        ("wd:" + pl.col("organism_wikidata").str.extract(r"(Q\d+)")).alias("wd_species")
    )

    lotus = lotus.select(
        [
            "wd_species",
            "organism_wikidata",
            "organism_name",
            "organism_taxonomy_01domain",
            "organism_taxonomy_02kingdom",
            "organism_taxonomy_03phylum",
            "organism_taxonomy_04class",
            "organism_taxonomy_05order",
            "organism_taxonomy_06family",
            "organism_taxonomy_08genus",
            "organism_taxonomy_09species",
            "organism_taxonomy_gbifid",
            "organism_taxonomy_ncbiid",
            "organism_taxonomy_ottid",
        ]
    )

    lotus = lotus.unique().to_pandas()
    lotus["wd_molecules"] = prediction
    graph = st.session_state["graph"]

    # get the ids of the molecules and the species
    species_phylo = st.session_state["species_phylo"]
    species_to_remove = list(set(lotus.wd_species) - set(species_phylo.node))
    lotus = lotus[~lotus.wd_species.isin(species_to_remove)]
    molecules_id = graph.get_node_ids_from_node_names(lotus.wd_molecules)
    species_id = graph.get_node_ids_from_node_names(lotus.wd_species)

    # create the sketching features
    sketching_features = st.session_state["sketching_features"]

    # create the sketching features
    pair_sketching = sketching_features.unknown(
        sources=molecules_id.astype("uint32"),
        destinations=species_id.astype("uint32"),
        feature_combination="addition",
    )

    # predict the probability of the link between the molecules and the species
    model = st.session_state["model"]
    out = model.predict_proba(pair_sketching)
    lotus["proba"] = out[:, 1]

    lotus["note"] = lotus.apply(
        lambda x: check_if_in_lotus(x.wd_species, x.wd_molecules, graph),
        axis=1,
    )
    final_df = lotus[
        (lotus.proba > threshold[0]) & (lotus.proba < threshold[1])
    ].sort_values("proba", ascending=False)

    st.write(final_df)

if prediction and selected_option == "Species":

    lotus = lotus.with_columns(
        ("wd:" + pl.col("structure_wikidata").str.extract(r"(Q\d+)")).alias(
            "wd_molecule"
        )
    )

    lotus = lotus.select(
        [
            "wd_molecule",
            "structure_wikidata",
            "structure_inchikey",
            "structure_smiles_2D",
            "structure_cid",
            "structure_taxonomy_npclassifier_01pathway",
            "structure_taxonomy_npclassifier_02superclass",
            "structure_taxonomy_npclassifier_03class",
            "structure_taxonomy_classyfire_01kingdom",
            "structure_taxonomy_classyfire_02superclass",
            "structure_taxonomy_classyfire_03class",
            "structure_taxonomy_classyfire_04directparent",
        ]
    )

    lotus = lotus.unique().to_pandas()
    lotus["wd_species"] = prediction
    graph = st.session_state["graph"]

    # get the ids of the molecules and the species
    species_phylo = st.session_state["species_phylo"]
    species_to_remove = list(set(lotus.wd_species) - set(species_phylo.node))
    lotus = lotus[~lotus.wd_species.isin(species_to_remove)]
    molecules_id = graph.get_node_ids_from_node_names(lotus.wd_molecule)
    species_id = graph.get_node_ids_from_node_names(lotus.wd_species)

    # create the sketching features
    sketching_features = st.session_state["sketching_features"]

    # create the sketching features
    pair_sketching = sketching_features.unknown(
        sources=molecules_id.astype("uint32"),
        destinations=species_id.astype("uint32"),
        feature_combination="addition",
    )

    # predict the probability of the link between the molecules and the species
    model = st.session_state["model"]
    out = model.predict_proba(pair_sketching)
    lotus["proba"] = out[:, 1]

    lotus["note"] = lotus.apply(
        lambda x: check_if_in_lotus(x.wd_species, x.wd_molecule, graph),
        axis=1,
    )
    final_df = lotus[
        (lotus.proba > threshold[0]) & (lotus.proba < threshold[1])
    ].sort_values("proba", ascending=False)

    st.write(final_df)
