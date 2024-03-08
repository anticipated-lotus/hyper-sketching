import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from grape import Graph

from src.models import LightGBM

st.set_page_config(
    page_title="Metabolite predictions",
    page_icon="🪷",
)

st.write("# Predicting LOTUS 🪷")

st.sidebar.success(
    "Select if you want to perform predictions on data that is already present in LOTUS or if you want to predict new data (slower)."
)

st.markdown(
    """
    TODO
"""
)

# Load the data
if "model" not in st.session_state:
    st.session_state["model"] = LightGBM.load_model("lightgbm_model.pkl")

if "graph" not in st.session_state:
    st.session_state["graph"] = Graph.from_csv(
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
        directed=False,
        load_edge_list_in_parallel=False,
        load_node_list_in_parallel=False,
    )

if "species_phylo" not in st.session_state:
    st.session_state["species_phylo"] = pd.read_csv(
        "./data/species/full_wikidata_taxonomy_nodes.csv"
    )

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
