from multiprocessing import Pool
import pandas as pd
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from requests import ConnectTimeout
from cache_decorator import Cache


def get_results(endpoint_url, query):
    """
    Executes a SPARQL query on the specified endpoint URL and returns the results in JSON format.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        query (str): The SPARQL query to execute.

    Returns:
        dict: The results of the SPARQL query in JSON format.
    """
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_taxonomy(species_as_wikidata_entity: str) -> pd.DataFrame:
    """
    Retrieves the taxonomy information for a given species from Wikidata.

    Args:
        species_as_wikidata_entity (str): The Wikidata entity ID of the species.

    Returns:
        pd.DataFrame: The taxonomy information as a pandas DataFrame.
    """
    endpoint_url = "https://query.wikidata.org/sparql"

    query = """PREFIX target: <http://www.wikidata.org/entity/%s>
    # Doctoral student/advisor network with a source from a spectific researcher
    PREFIX gas: <http://www.bigdata.com/rdf/gas#>

    SELECT DISTINCT ?taxon ?taxonLabel ?relative ?relativeLabel ?depth
    WHERE {
      { 
        SELECT ?taxon ?relative (MIN(?depth1) as ?depth)
        WHERE {
          SERVICE gas:service {
            gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.BFS" ;
                    gas:in target: ;
                    gas:traversalDirection "Forward" ;
                    gas:out ?taxon ;
                    gas:out1 ?depth1 ;
                    gas:out2 ?relative ;
                    gas:linkType wdt:P171 ;
          }
        }
        GROUP BY ?taxon ?relative
      }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en,da,sv,jp,zh,ru,fr,de" .  } 
    }

    """ % (
        species_as_wikidata_entity
    )
    try:
        results = get_results(endpoint_url, query)
        return pd.json_normalize(results["results"]["bindings"])

    except ConnectionAbortedError:
        time.sleep(0.5)
        results = get_results(endpoint_url, query)
        return pd.json_normalize(results["results"]["bindings"])

    except ConnectionError:
        time.sleep(0.5)
        results = get_results(endpoint_url, query)
        return pd.json_normalize(results["results"]["bindings"])

    except ConnectTimeout:
        time.sleep(0.5)
        results = get_results(endpoint_url, query)
        return pd.json_normalize(results["results"]["bindings"])


def get_all_species_in_lotus():
    """
    Retrieves all species related to Lotus from Wikidata.

    Returns:
        pd.DataFrame: The species information as a pandas DataFrame.
    """
    endpoint_url = "https://query.wikidata.org/sparql"

    query = """#!WDDEFAULTIMPORTS
    SELECT DISTINCT ?taxon ?taxon_name ?taxon_ncbi ?taxon_gbif ?taxon_otl WHERE {
        ?compound wdt:P235 ?compound_inchikey;
                 p:P703 [
                   ps:P703 ?taxon ;
                           prov:wasDerivedFrom / pr:P248 [
                             wdt:P356 ?reference_doi
                           ]
                 ] .
      OPTIONAL { ?taxon wdt:P225 ?taxon_name. }
      OPTIONAL { ?taxon wdt:P685 ?taxon_ncbi. }
      OPTIONAL { ?taxon wdt:P846 ?taxon_gbif. }
      OPTIONAL { ?taxon wdt:P9157 ?taxon_otl. }
    }"""
    results = get_results(endpoint_url, query)
    return (
        pd.json_normalize(results["results"]["bindings"])
        .drop_duplicates(subset="taxon.value")
        .reset_index(drop=True)
    )


def convert_to_edges(df: pd.DataFrame):
    """
    Converts a DataFrame of taxonomy information into edges.

    Args:
        df (pd.DataFrame): The DataFrame containing taxonomy information.

    Returns:
        pd.DataFrame: The edges of the taxonomy graph as a pandas DataFrame.
    """
    if "taxon.value" not in df.columns:
        return pd.DataFrame({0: [float("nan")], 1: [float("nan")]})

    parent = df["taxon.value"].copy()
    child = df["relative.value	"].copy()
    return pd.DataFrame({0: child, 1: parent}).dropna().reset_index(drop=True)


@Cache(
    cache_dir="data/species/all_species/{species}/",
    cache_path="{cache_dir}/{_hash}.csv.gz",
)
def taxonomy_in_edges(species: str):
    """
    Retrieves the taxonomy information for a single species and converts it into edges.

    Args:
        species (str): The Wikidata entity ID of the species.

    Returns:
        pd.DataFrame: The edges of the taxonomy graph for the species as a pandas DataFrame.
    """
    taxo = get_taxonomy(species)
    out = convert_to_edges(taxo)
    return pd.DataFrame(out)


# def get_taxo_in_edges_all(all_species, n_cpus=4):
#    """
#    Retrieves the taxonomy information for all species and converts them into edges.
#
#    Args:
#        all_species (list): A list of Wikidata entity IDs of the species.
#        n_cpus (int): The number of CPUs to use for parallel processing.
#
#    Returns:
#        pd.DataFrame: The edges of the taxonomy graph for all species as a pandas DataFrame.
#    """
#    with Pool(processes=n_cpus) as pool:
#        ls = pool.map(taxonomy_in_edges, all_species)
#    return pd.concat(ls).drop_duplicates().reset_index(drop=True).dropna()


if __name__ == "__main__":
    x = get_taxonomy("wd:Q7224923")
    # the last WD entity (Q15869) is not a species so if there are this kind of errors, the function will just ignore those
    ls = ["wd:Q7224923", "wd:Q158695"]
    y = [taxonomy_in_edges(i) for i in ls]
    y = pd.concat(y).drop_duplicates().reset_index(drop=True).dropna()
    print(y)
