import multiprocessing
from functools import partial
from typing import Dict, Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


def calculate_metrics(
    label: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    assert isinstance(label, np.ndarray)
    assert isinstance(prediction, np.ndarray)
    assert label.shape == prediction.shape
    thresholded_prediction = prediction > 0.5
    return {
        "accuracy": accuracy_score(label, thresholded_prediction),
        "balanced_accuracy": balanced_accuracy_score(label, thresholded_prediction),
        "f1": f1_score(label, thresholded_prediction),
        "recall": recall_score(label, thresholded_prediction),
        "auroc": roc_auc_score(label, prediction),
        "auprc": average_precision_score(label, prediction),
        "mcc": matthews_corrcoef(label, thresholded_prediction),
    }


# Fingerprint generation
# (@ming if we want to use inchi as an input, inchi should be changed to SMILES and the SMILES should be standardized)
def calculate_fingerprint(smiles, radi):
    binary = np.zeros((2048 * (radi)), int)
    formula = np.zeros((2048), int)
    mol = Chem.MolFromSmiles(smiles)

    mol = Chem.AddHs(mol)
    mol_bi = {}
    for r in range(radi + 1):
        mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=r, bitInfo=mol_bi, nBits=2048
        )
        mol_bi_QC = []
        for i in mol_fp.GetOnBits():
            num_ = len(mol_bi[i])
            for j in range(num_):
                if mol_bi[i][j][1] == r:
                    mol_bi_QC.append(i)
                    break

        if r == 0:
            for i in mol_bi_QC:
                formula[i] = len([k for k in mol_bi[i] if k[1] == 0])
        else:
            for i in mol_bi_QC:
                binary[(2048 * (r - 1)) + i] = len([k for k in mol_bi[i] if k[1] == r])

    return np.concatenate(
        [
            formula.reshape(-1),
            binary.reshape(-1),
        ]
    )


def calculate_fingerprint_parallel(smiles: Iterable, radi) -> list:
    pool = multiprocessing.Pool()
    return list(
        tqdm(
            pool.imap(
                partial(calculate_fingerprint, radi=2),
                smiles,
                chunksize=1000,
            ),
        )
    )
