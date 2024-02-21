from src.models import XGBoost
import pandas as pd
import numpy as np
from grape import Graph
from cache_decorator import Cache
from typing import Type
from .models.abstract_model import AbstractModel


@Cache(
    cache_dir="predictions/species/{species}/",
    cache_path="{cache_dir}/{_hash}.csv.gz",
    args_to_ignore=["sketching_features"],
    use_approximated_hash=True,
)
def predict_all_molecules_for_one_species(
    species: str,
    graph: Graph,
    lotus: pd.DataFrame,
    model: Type[AbstractModel],
    sketching_features,
) -> pd.DataFrame:
    """
    This function takes a species and returns a dataframe with the molecules that are predicted to be linked to the species.
    Args:
    - species (str): the species of interest
    - graph (Graph): the graph object
    - lotus (pd.DataFrame): the LOTUS database
    - model (Type[AbstractModel]): the model to use for the predictions
    - sketching_features: the sketching features to use for the predictions
    Returns:
    - pd.DataFrame: a dataframe with the molecules that are predicted to be linked to the species
    """
    # check if the species is in the LOTUS database
    if species not in lotus.wd_species.unique():
        raise ValueError(f"The species {species} is not in the LOTUS database.")

    # create a dataframe with all the molecules and the species of interest
    df = pd.DataFrame(
        {
            "molecule_wikidata": [i for i in sorted(lotus.wd_molecule.unique())],
        }
    )
    df["species_wikidata"] = species
    df["species_name"] = lotus[lotus.wd_species == species].organism_name.unique()[0]

    # get the ids of the molecules and the species
    molecules_id = graph.get_node_ids_from_node_names(df.molecule_wikidata)
    species_id = graph.get_node_ids_from_node_names(df.species_wikidata)

    # create the sketching features
    pair_sketching = sketching_features.unknown(
        sources=molecules_id.astype("uint32"),
        destinations=species_id.astype("uint32"),
        feature_combination="addition",
    )

    # predict the probability of the link between the molecules and the species
    out = model.predict_proba(pair_sketching)
    df["proba"] = out[:, 1]

    df["note"] = df.apply(
        lambda x: check_if_in_lotus(x.species_wikidata, x.molecule_wikidata, graph),
        axis=1,
    )

    final_df = df[df.proba > 0.75].sort_values("proba", ascending=False)
    return final_df


@Cache(
    cache_dir="predictions/molecules/{molecule}/",
    cache_path="{cache_dir}/{_hash}.csv.gz",
    args_to_ignore=["sketching_features"],
    use_approximated_hash=True,
)
def predict_all_species_for_one_molecule(
    molecule: str,
    graph: Graph,
    lotus: pd.DataFrame,
    model: Type[AbstractModel],
    sketching_features,
) -> pd.DataFrame:
    """
    This function takes a molecule and returns a dataframe with the species that are predicted to be linked to the molecule.
    Args:
    - molecule (str): the molecule of interest
    - graph (Graph): the graph object
    - lotus (pd.DataFrame): the LOTUS database
    - model (Type[AbstractModel]): the model to use for the predictions
    - sketching_features: the sketching features to use for the predictions
    Returns:
    - pd.DataFrame: a dataframe with the species that are predicted to be linked to the molecule
    """

    # check if the molecule is in the LOTUS database
    if molecule not in lotus.wd_molecule.unique():
        raise ValueError(f"The molecule {molecule} is not in the LOTUS database.")

    # create a dataframe with all the species and the molecule of interest
    df = pd.DataFrame(
        {
            "species_wikidata": [i for i in sorted(lotus.wd_species.unique())],
        }
    )
    df["molecule_wikidata"] = molecule
    df["molecule_inchikey"] = lotus[
        lotus.wd_molecule == molecule
    ].structure_inchikey.unique()[0]

    # get the ids of the molecules and the species
    molecules_id = graph.get_node_ids_from_node_names(df.molecule_wikidata)
    species_id = graph.get_node_ids_from_node_names(df.species_wikidata)

    # create the sketching features
    pair_sketching = sketching_features.unknown(
        sources=molecules_id.astype("uint32"),
        destinations=species_id.astype("uint32"),
        feature_combination="addition",
    )

    # predict the probability of the link between the molecules and the species
    out = model.predict_proba(pair_sketching)
    df["proba"] = out[:, 1]

    df["note"] = df.apply(
        lambda x: check_if_in_lotus(x.species_wikidata, x.molecule_wikidata, graph),
        axis=1,
    )

    final_df = df[df.proba > 0.75].sort_values("proba", ascending=False)
    return final_df


# apply a function that looks at the species, anf if the molecule id is in the neighborhood of the species,
# then it should write in a new column called "note" this : "The link between this molecules and this species is already in LOTUS."
def check_if_in_lotus(species, molecule, graph: Graph):
    if molecule in graph.get_neighbour_node_names_from_node_name(species):
        return "This link is already in LOTUS."
    return np.nan
