import sys
import time
from io import StringIO

import pandas as pd
from cache_decorator import Cache
from requests import ConnectTimeout, request
from SPARQLWrapper import JSON, SPARQLWrapper


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


def get_taxonomy_of_species(species_as_wikidata_entity: str) -> pd.DataFrame:
    """
    Retrieves the taxonomy information for a given species from Wikidata.

    Args:
        species_as_wikidata_entity (str): The Wikidata entity ID of the species (should start with Q and some numbers).

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
    child = df["relative.value"].copy()
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
    taxo = get_taxonomy_of_species(species)
    out = convert_to_edges(taxo)
    return pd.DataFrame(out)


def get_full_taxonomy_of_wikidata():
    endpoint_url = "https://query.wikidata.org/sparql"

    query = """PREFIX hint: <http://www.bigdata.com/queryHints#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?taxon ?parent {
    ?taxon wdt:P171 ?parent. hint:Prior hint:rangeSafe true.
    }
    """
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


def get_class_or_subclass_of_chemical_structures():
    """
    This query was inspired by a query from Adriano Rutz. See: https://adafede.github.io/sparql/queries/wd_chemicals_classes.html

    Retrieves the class or subclass of the chemical structures in Wikidata.

    Returns as dataframe with 3 columns : the instance, the class and the structure.
    The instance is the instance of the `class`. The class is the parent node and the structure is the child node.
    Returns:
        pd.DataFrame: The class or subclass of the chemical structures as a pandas DataFrame.
    """
    method = "GET"
    QLEVER_URL = "https://qlever.cs.uni-freiburg.de/api/wikidata"
    query = """#+ summary: Chemicals subclass of chemical class
    #+ description: Chemicals subclass of chemical class
    #+ endpoint: https://query.wikidata.org/bigdata/namespace/wdq/sparql
    #+ pagination: 100
    #+ method: GET
    #+ tags:
    #+   - Chemicals
    #+   - Chemical classes
    
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX hint: <http://www.bigdata.com/queryHints#>
    #title: Chemicals subclass of chemical class
    SELECT ?instance ?structure ?class WHERE {
        VALUES ?instance {
            wd:Q15711994 # group of isomeric entities
            wd:Q17339814 # group or class of chemical substances
            wd:Q47154513 # structural class of chemical entities
            wd:Q55640599 # group of chemical entities
            wd:Q56256173 # class of chemical entities with similar applications or functions
            wd:Q56256178 # class of chemical entities with similar source or occurrence
            wd:Q55662456 # group of ortho, meta, para isomers
            wd:Q59199015 # group of stereoisomers
            wd:Q72070508 # group or class of chemical entities
            wd:Q74892521 # imprecise class of chemical entities
        }
        ?class wdt:P31 ?instance.
        ?structure wdt:P279 ?class.
    }"""

    out = request(
        method,
        QLEVER_URL,
        params={"query": query},
        headers={
            "Accept": "text/csv",
            "Accept-Encoding": "gzip,deflate",
            "User-Agent": "LOTUS project database dumper",
        },
        timeout=70,
    ).text

    return pd.read_csv(StringIO(out))


def get_wikidata_chemical_class_taxonomy():
    """
    This query was inspired by a query from Adriano Rutz. See: https://adafede.github.io/sparql/queries/wd_chemical_classes_taxonomy.html

    Retrieves the chemical class taxonomy of Wikidata.

    Returns as dataframe with 3 columns : the instance, the class and the parent.
    The instance is the instance of the `class` (child node). The class is the child node and the parent is the parent node.
    """
    method = "GET"
    QLEVER_URL = "https://qlever.cs.uni-freiburg.de/api/wikidata"

    query = """
    #+ summary: Chemical classes taxonomy
    #+ description: Chemical classes taxonomy
    #+ endpoint: https://query.wikidata.org/bigdata/namespace/wdq/sparql
    #+ pagination: 100
    #+ method: GET
    #+ tags:
    #+   - Chemical classes
    
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX hint: <http://www.bigdata.com/queryHints#>
    #title: Chemical classes taxonomy
    SELECT * WHERE {
        VALUES ?instance {
            wd:Q15711994 # group of isomeric entities
            wd:Q17339814 # group or class of chemical substances
            wd:Q47154513 # structural class of chemical entities
            wd:Q55640599 # group of chemical entities
            wd:Q56256173 # class of chemical entities with similar applications or functions
            wd:Q56256178 # class of chemical entities with similar source or occurrence
            wd:Q55662456 # group of ortho, meta, para isomers
            wd:Q59199015 # group of stereoisomers
            wd:Q72070508 # group or class of chemical entities
            wd:Q74892521 # imprecise class of chemical entities
        }
        ?class wdt:P31 ?instance.
        OPTIONAL { ?class wdt:P279 ?parent. }
    }"""
    out = request(
        method,
        QLEVER_URL,
        params={"query": query},
        headers={
            "Accept": "text/csv",
            "Accept-Encoding": "gzip,deflate",
            "User-Agent": "LOTUS project database dumper",
        },
        timeout=70,
    ).text

    return pd.read_csv(StringIO(out))
