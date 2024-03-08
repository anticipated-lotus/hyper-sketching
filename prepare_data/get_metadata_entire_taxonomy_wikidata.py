# This script was inspired from https://github.com/lotusnprod/lotus-search/blob/0e07c77e954359c7e1bae9df0952b7d12c02d697/update/taxo_helper.py from the https://github.com/lotusnprod/lotus-search repo
# Check over there for the original (and maybe now) alternative version

import csv
import logging
import os
import sys
from collections import defaultdict, deque
from io import StringIO
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from requests import request
from tqdm import tqdm

# Set the URL constant
QLEVER_URL = "https://qlever.cs.uni-freiburg.de/api/wikidata"

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)


def convert_to_int_safe(s: str) -> int | None:
    try:
        result = int(s)
        return result
    except ValueError:
        logging.error(f"{s} is not a valid integer.")
        return None


def generate_taxon_parents_with_distance(path: Path) -> list[tuple[int, int, int]]:
    graph = defaultdict(list)
    distances = []
    with open(path / "full_wikidata_taxonomy_edges.csv", "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        taxon_index = headers.index("child")
        parent_index = headers.index("parent")

        for row in reader:
            taxon_id = row[taxon_index]
            parent_id = row[parent_index]

            if taxon_id is None or parent_id is None:
                continue
            graph[taxon_id].append(parent_id)
    # Good ol' BFS
    for node in tqdm(list(graph.keys())):
        visited = {node: 0}
        queue = deque([node])
        while queue:
            current_node = queue.popleft()
            current_distance = visited[current_node]

            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited[neighbor] = current_distance + 1
                    distances.append((node, neighbor, current_distance + 1))

    return distances


def sparql_to_csv(query: str, url: str = QLEVER_URL, as_post: bool = False) -> str:
    method = "POST" if as_post else "GET"
    return request(
        method,
        url,
        params={"query": query},
        headers={
            "Accept": "text/csv",
            "Accept-Encoding": "gzip,deflate",
            "User-Agent": "LOTUS project database dumper",
        },
        timeout=70,
    ).text


def main():
    # First we genereate a list of tuples containing the taxon, parent and distance to the parent.
    # This creates a very big dataframe of 85'000'000 rows
    ls = generate_taxon_parents_with_distance(Path("./data/species"))
    taxon_to_parents = pl.DataFrame(ls)
    del ls

    # Then we create a dataframe with the taxon and the direct parent only
    query = """
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        SELECT ?taxon ?taxon_name ?taxon_rank ?taxon_rank_label ?taxon_parent ?parent_name WHERE {
        {
            ?taxon wdt:P225 ?taxon_name;
                   wdt:P105 ?taxon_rank;
        		   wdt:P171 ?taxon_parent.
        		   }
        		   {?taxon_rank rdfs:label ?taxon_rank_label.
          FILTER (lang(?taxon_rank_label) = "en")
        }
        		   {?taxon_parent wdt:P225 ?parent_name.
        }
        }
    """
    # We use the QLEVER API to get the results of the query
    out = sparql_to_csv(query)
    taxon_to_direct_parent_only = pl.read_csv(StringIO(out))
    del out

    # We add the "wd:" prefix to the taxon and taxon_parent columns
    # this will allow to merge the two dataframes
    taxon_to_direct_parent_only = taxon_to_direct_parent_only.with_columns(
        ("wd:" + pl.col("taxon").str.extract(r"(Q\d+)")).alias("taxon_wd")
    )

    # we then merge the two dataframes on the taxon_wd column
    # this will allow us to get the taxon_name and taxon_rank_label for each taxon
    # This results in a dataframe with 85'000'000 rows with taxon-parent and metadata of the parent of the taxon
    taxon_to_parents_with_metadata = taxon_to_parents.join(
        taxon_to_direct_parent_only,
        left_on="column_1",
        right_on="taxon_wd",
    )

    # Now we want to pivot the table so that we have the taxon name as values, the different ranks
    # as columns and the taxon of interest as index.
    out = taxon_to_parents_with_metadata.pivot(
        values="taxon_name",
        index="column_0",
        columns="taxon_rank_label",
        aggregate_function="first",
    )

    # We repeat the same process for the taxon itself.
    # We will then have the metadata of the taxon (first column in the big dataframe)
    out_2 = taxon_to_parents.join(
        taxon_to_direct_parent_only,
        left_on="column_0",  # here we changed from column_1 to column_0
        right_on="taxon_wd",
    ).pivot(
        values="taxon_name",
        index="column_0",
        columns="taxon_rank_label",
        aggregate_function="first",
    )

    # We then merge the two dataframes on the taxon column so then we are able to gave all the metadata needed
    merged_df = out.join(
        out_2,
        on="column_0",
        how="outer",
    )
    del out
    del out_2

    # Since now we have a dataframe with the double of the columns we need to remove the columns that end with _right
    # However some columns ending in "_right" are not null and contain some information about the taxon so we want to
    # keep them.
    right_columns = [col for col in merged_df.columns if col.endswith("_right")]

    for right_col in right_columns:
        # Extract the base column name (without '_right')
        base_col = right_col[:-6]

        # Use 'over' to select the non-null values from the base column and the '_right' column
        merged_df = merged_df.with_columns(
            merged_df.select(
                test=pl.when(pl.col(base_col).is_not_null())
                .then(pl.col(base_col))
                .otherwise(pl.col(right_col)),
            ).rename({"test": base_col})
        )

    merged_df = merged_df.drop(right_columns)

    # Finally we only keep the columns that usually contain the most values
    # and we rename the first column to "node".
    # This output resembles to the metadata of the taxon we have in LOTUS
    final_output = merged_df.select(
        [
            "column_0",
            "superkingdom",
            "domain",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
            "subspecies",
        ]
    ).rename({"column_0": "node"})

    final_output.write_csv("./data/species/full_taxonomy_metadata.csv.gz")


if __name__ == "__main__":
    main()
