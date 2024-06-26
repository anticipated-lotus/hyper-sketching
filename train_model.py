# # Let's implement the predictions task for species
import numpy as np
from ensmallen import HyperSketchingPy
from grape import Graph

from src.models import LightGBM, XGBoost


def main():
    graph = Graph.from_csv(
        name="full_graph_clean",
        node_path="./data/full_graph_clean_nodes.csv",
        edge_path="./data/full_graph_clean_edges.csv",
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
    print("The graph hash is : ", graph.hash())

    # We then want to filter the graph so that we only keep the edges "biolink:in_taxon"

    pos = graph.filter_from_names(
        edge_type_names_to_keep=["biolink:in_taxon"],
    )

    # The graph that we juste generated is then the positive edges. We now need to sample the negative edges:

    neg = pos.sample_negative_graph(
        number_of_negative_samples=pos.get_number_of_directed_edges(),
        sample_edge_types=False,
        only_from_same_component=False,
        use_scale_free_distribution=True,
        random_state=23391 * (3 + 1),
    )

    # Next we create the features:

    number_of_hops = 2
    normalize = False
    combination = "addition"

    print("Creating the features...")
    sketching_features = HyperSketchingPy(
        hops=number_of_hops, normalize=normalize, graph=graph
    )
    sketching_features.fit()

    # sketching for positive training edges
    pos_sources = pos.get_directed_source_node_ids()
    pos_destinations = pos.get_directed_destination_node_ids()

    print("Creating the positive features...")
    sk_positive_features = sketching_features.positive(
        sources=pos_sources,
        destinations=pos_destinations,
        feature_combination=combination,
    )

    # sketching for training negatives
    print("Creating the negative features...")
    neg_sources = neg.get_directed_source_node_ids()
    neg_destinations = neg.get_directed_destination_node_ids()
    sk_negative_features = sketching_features.negative(
        sources=neg_sources,
        destinations=neg_destinations,
        feature_combination=combination,
    )

    # We can then create the label and the data for the training of our model

    X = np.concatenate([sk_positive_features, sk_negative_features])
    label_pos = np.ones(sk_positive_features.shape[0])
    label_neg = np.zeros(sk_negative_features.shape[0])
    label = np.concatenate([label_pos, label_neg])

    # randomize the order of the training data
    random_state = 43
    indices = np.arange(X.shape[0])
    rnd = np.random.RandomState(random_state)
    rnd.shuffle(indices)
    X_shuffled = X[indices]

    label_shuffled = label[indices]

    # Let's create the model now :

    model = LightGBM(
        boosting_type="gbdt",
        importance_type="gain",
        max_depth=20,
        n_estimators=200,
        num_leaves=51,
    )

    print("Training the model...")
    model.fit(X_shuffled, label_shuffled)
    model.dump_model("lightgbm_model_new.pkl")


if __name__ == "__main__":
    main()
