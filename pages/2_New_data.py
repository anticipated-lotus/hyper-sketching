from time import sleep
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import polars as pl
import requests
import streamlit as st
from ensmallen import HyperSketchingPy
from grape import Graph

from src.models import XGBoost
from src.molecules import convert_to_csv, filter_df, get_result, structure_query

# Constants
URL_CLASSYFIRE = "https://structure.gnps2.org/classyfire?smiles="
URL_NP_CLASSIFIER = "https://npclassifier.gnps2.org/classify?smiles="

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


# Functions
def classify_with_classyifre(compound: str) -> dict:
    """Submit a compound information to the ClassyFire service for evaluation
    and receive the classification of the compound.

    :param compound: The SMILES of the compound of interest
    :type compound: str
    :return: A dictionary with the results of the classification
    :rtype: dict

    >>> classify_with_classyifre('CCC')

    """
    r = requests.get(
        URL_CLASSYFIRE + compound,
    )
    r.raise_for_status()
    return r.json()


def classify_with_np_classifier(compound: str) -> int:
    """Submit a compound information to the NP Classifier service for evaluation
    and receive the classification of the compound.

    :param compound: The SMILES of the compound of interest
    :type compound: str
    :return: A dictionary with the results of the classification
    :rtype: dict

    >>> classify_with_np_classifier('CCC')

    """
    r = requests.get(
        URL_NP_CLASSIFIER + compound,
    )
    r.raise_for_status()
    return r.json()


# apply a function that looks at the species, anf if the molecule id is in the neighborhood of the species,
# then it should write in a new column called "note" this : "The link between this molecules and this species is already in LOTUS."
def check_if_in_lotus(species, molecule, graph: Graph):
    if molecule in graph.get_neighbour_node_names_from_node_name(species):
        return "This link is already in LOTUS."
    return np.nan


compound = st.text_input(
    """ Please enter the SMILES of the compound of interest. 
    """,
    value="",
)

# Now that the user has entered the compound, we can classify it
if compound:
    # we start to classify the compound with the NP Classifier
    dct = classify_with_np_classifier(compound)
    _ = dct.pop("isglycoside")

    # We first create the edges dataframe
    edges_np_classifier = (
        pd.concat(
            [
                pd.DataFrame([compound]),
                pd.DataFrame.from_dict(dct, orient="index"),
            ]
        )
        .dropna()
        .reset_index(drop=True)
    )

    edges_np_classifier[1] = edges_np_classifier.iloc[:, 0].shift(-1)
    edges_np_classifier.dropna(inplace=True)
    edges_np_classifier.rename(columns={0: "child", 1: "parent"}, inplace=True)
    st.write("NP Classifier classified this molecule as : ")
    st.write(edges_np_classifier)
    edges_np_classifier["type"] = "biolink:subclass_of"

    # then the nodes dataframe
    nodes_np_classifier = (
        pd.DataFrame(
            {
                "node": pd.concat(
                    [edges_np_classifier.child, edges_np_classifier.parent]
                ),
                "type": "biolink:ChemicalEntity",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # we then classify the compound with ClassyFire
    # We first create the edges dataframe
    query_id = structure_query(compound)
    with st.spinner(
        "Please while the molecule is being classified on ClassyFire servers. This should not take more than a minute."
    ):
        sleep(10)
    st.success("Done!")
    edges_classyfire = filter_df(convert_to_csv(get_result(query_id)))
    st.write("ClassyFire classified this molecule as : ")
    st.write(edges_classyfire["Result"])
    edges_classyfire.drop(
        columns=[
            "CompoundID",
            "ClassifiedResults",
            "Classification",
            "Result",
        ],
        inplace=True,
    )
    edges_classyfire = pd.DataFrame(
        pd.concat(
            [
                edges_classyfire["ChemOntID"],
                pd.Series(compound),
            ]
        )[::-1]
    ).reset_index(drop=True)

    edges_classyfire[1] = edges_classyfire.iloc[:, 0].shift(-1)
    edges_classyfire.dropna(inplace=True)
    edges_classyfire["type"] = "biolink:subclass_of"
    edges_classyfire.rename(
        columns={
            0: "child",
            1: "parent",
        },
        inplace=True,
    )

    # then the nodes dataframe
    nodes_classyfire = (
        pd.DataFrame(
            {
                "node": pd.concat([edges_classyfire.child, edges_classyfire.parent]),
                "type": "biolink:ChemicalEntity",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

if compound:
    st.session_state["graph_classyfire"] = Graph.from_pd(
        directed=False,
        edges_df=edges_classyfire,
        nodes_df=nodes_classyfire,
        node_name_column="node",
        node_type_column="type",
        edge_src_column="child",
        edge_dst_column="parent",
        edge_type_column="type",
    )

if compound:
    st.session_state["graph_np_classifier"] = Graph.from_pd(
        directed=False,
        edges_df=edges_np_classifier,
        nodes_df=nodes_np_classifier,
        node_name_column="node",
        node_type_column="type",
        edge_src_column="child",
        edge_dst_column="parent",
        edge_type_column="type",
    )

if compound:
    st.session_state["graph_merged"] = (
        st.session_state["graph"]
        | st.session_state["graph_classyfire"]
        | st.session_state["graph_np_classifier"]
    )


lotus = st.session_state["lotus"]
threshold = st.slider(
    "Select the threshold for the predictions",
    min_value=0.0,
    max_value=1.0,
    value=(0.75, 1.0),
    step=0.01,
)

if compound:
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
    lotus["molecule"] = compound
    graph = st.session_state["graph_merged"]

    # get the ids of the molecules and the species
    species_phylo = st.session_state["species_phylo"]
    species_to_remove = list(set(lotus.wd_species) - set(species_phylo.node))
    lotus = lotus[~lotus.wd_species.isin(species_to_remove)]
    molecules_id = graph.get_node_ids_from_node_names(lotus.molecule)
    species_id = graph.get_node_ids_from_node_names(lotus.wd_species)

    # create the sketching features
    sketching_features = HyperSketchingPy(
        hops=2,
        normalize=False,
        graph=graph,
    )
    sketching_features.fit()

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
        lambda x: check_if_in_lotus(x.wd_species, x.molecule, graph),
        axis=1,
    )
    final_df = lotus[
        (lotus.proba > threshold[0]) & (lotus.proba < threshold[1])
    ].sort_values("proba", ascending=False)

    st.write(final_df)
