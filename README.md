[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
# LOTUS anticipated

## Prepare environment 
### Install pyenv
First you will need to download `pyenv`  and `pipx`:
```bash
curl https://pyenv.run | bash
```
or use Homebrew:
```bash
brew install pyenv
```

### Install pipx
The you can install pipx:
```bash
brew install pipx
pipx ensurepath
```
Or if you are on Linux:
```bash
sudo apt update
sudo apt install pipx
pipx ensurepath
```

### Install poetry
Then you can install poetry:
```bash
pipx install poetry
```

Finally you can install the environment:
```bash
poetry install
```


### Alternative: Install the environment in conda (NOT recommended):
```bash 
conda env create --file environment.yml
conda activate grape
```

## Download preprocessed data
For running the analysis you can download the data from Zenodo: TODO
```bash
TODO
``` 

### Run the analysis
You may choose which ever model you want from the [`src/models`](https://github.com/mvisani/anticipated_lotus/tree/main/src/models) folder. If you want you can also implement your own model. The available models are : 
- [DecisionTree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit)
- [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)

Before running the script, you should change the variable `max_evals` in the `run_decision_tree.py` file to 100 or more.

## Prepare the data locally

First thing to do is to download LOTUS from Zenodo: 
```bash
wget https://zenodo.org/record/7534071/files/230106_frozen_metadata.csv.gz
mv ./230106_frozen_metadata.csv.gz ./data/molecules/230106_frozen_metadata.csv.gz
wget http://classyfire.wishartlab.com/system/downloads/1_0/chemont/ChemOnt_2_1.obo.zip
mv ./ChemOnt_2_1.obo.zip ./data/molecules/ChemOnt_2_1.obo.zip
# unzip the file
unzip ./data/molecules/ChemOnt_2_1.obo.zip
mv ./ChemOnt_2_1.obo ./data/molecules/ChemOnt_2_1.obo

wget https://raw.githubusercontent.com/mwang87/NP-Classifier/master/Classifier/dict/index_v1.json
mv ./index_v1.json ./data/molecules/NPClassifier_index.json
```



### Prepare molecules
First run the following command to prepare lotus, the molecules and the molecule ontology:
```bash
python prepare_data/prepare_lotus.py
python prepare_data/prepare_mol_to_chemont.py
python prepare_data/prepare_NPClassifier.py
```

### Prepare species
The species edge list will take a bit longer to prepare (2-3 minutes). We are downloading the entire taxonomy from Wikidata using [this query](https://w.wiki/9FKC).
```bash
python prepare_data/prepare_species.py 
```

### Prepare graph
Now we need to prepare the data to be suitable for [grape](https://github.com/AnacletoLAB/grape) library. 
```bash
python prepare_data/prepare_graph.py
```

Finally we need to merge the data from NCBI and LOTUS. 
```bash
python prepare_data/prepare_merge_ncbi.py
```

Once this is done you should have in the `data` folder the following structure, with the following graphs available:
```shell
.
├── full_graph_with_ncbi_clean_edges.csv
├── full_graph_with_ncbi_clean_nodes.csv
├── full_graph_with_ncbi_edges.csv
├── full_graph_with_ncbi_nodes.csv
├── full_wd_taxonomy_with_molecules_in_lotus_clean_edges.csv
├── full_wd_taxonomy_with_molecules_in_lotus_clean_nodes.csv
├── full_wd_taxonomy_with_molecules_in_lotus_edges.csv
├── full_wd_taxonomy_with_molecules_in_lotus_nodes.csv
├── lotus
│   ├── lotus_edges.csv
│   └── lotus_nodes.csv
├── molecules
│   ├── 230106_frozen_metadata.csv.gz
│   ├── ChemOnt_2_1.obo
│   ├── ChemOnt_2_1.obo.zip
│   ├── NPClassifier_index.json
│   ├── chemont_edges.csv
│   ├── chemont_nodes.csv
│   ├── mol_to_chemont_edges.csv
│   ├── mol_to_chemont_nodes.csv
│   ├── mol_to_np_edges.csv
│   └── mol_to_np_nodes.csv
└── species
    ├── full_wikidata_taxonomy_edges.csv
    └── full_wikidata_taxonomy_nodes.csv
```

### Choose the graph
You can choose which graph you want to use for the analysis. Here is an explanation of the different graphs:
- `full_graph_with_ncbi_clean` : This graph contains all the data from LOTUS, the taxonomy from wikidata and the taxonomy from NCBI. It is cleaned meaning that there are no disconnected components in the graph.
- `full_graph_with_ncbi`: This graph contains all the data from LOTUS, the taxonomy from wikidata and the taxonomy from NCBI. This one is **not** cleaned meaning that there are some disconnected components in the graph.
- `full_wd_taxonomy_with_molecules_in_lotus_clean` : This graph contains the taxonomy from wikidata and the molecules from LOTUS (with the classification of the molecules). It is cleaned meaning that there are no disconnected components in the graph.
- `full_wd_taxonomy_with_molecules_in_lotus` : This graph is the same as the previous one. This one is **not** cleaned meaning that there are some disconnected components in the graph.
- `lotus/lotus` : This graph is only a bipartite graph with the species and the molecules from LOTUS.
- `molecules/chemont` : This graph contains only the different classes of molecules from Classyfire.
- `molecules/mol_to_chemont` : This graph contains the molecules **and** the classes of molecules from Classyfire.
- `molecules/mol_to_np` : This graph contains the molecules **and** the classes of molecules from NPClassifier.
- `species/full_wikidata_taxonomy` : This graph contains the entire taxonomy of the species on Earth from Wikidata.


In our case we will either use the `full_graph_with_ncbi_clean` or `full_wd_taxonomy_with_molecules_in_lotus_clean`. Further tests need to be made to see which one is the best for the predictions.

### Run the analysis
This is not possible at the moment because the module `ensmallen` from [grape](https://github.com/AnacletoLAB/grape) does not support the HyperSketching yet. Once it will be available, we recommend to first run the `run_model_dummy.py` script with `max_eval=1`. This will first create the sketching of the different holdouts of training and testing. Then you can run the script with `max_eval=100` or more to find the best parameters of the model.

### Train the model
Once the best parameters are found, you can train the model using the `train_model.py` script. You should manually change the parameters and the model in the script according to the best parameters found.


## Streamlit app
Here are some new molecules from :
* `CCC\C=C\C=C\C=C\C(O)CC1(O)OC(CC(O)C(O)C(O)C2OC(=O)C(C)C2O)C(O)C(O)C1O` from https://pubs.acs.org/doi/10.1021/acs.jnatprod.3c01043
* `CC(C)CCC(C)C(=O)NCCCNC(=N)N` from https://pubs.acs.org/doi/10.1021/acs.jnatprod.3c01186
* `CC2=CC(=O)CC3C(C)C(C)(COC1CC(O)C(O)C(C)O1)CCC23C` from https://pubs.acs.org/doi/10.1021/acs.jnatprod.3c00752
* `CC1CCc2c(CCC=C(C)C)cnc3c(C)cc(O)c1c23` from https://pubs.acs.org/doi/10.1021/acs.jnatprod.3c01072
