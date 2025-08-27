import json
import os
from Bio import Phylo
import numpy as np

from ..distance import j_one


def get_patient_tree(patient: str) -> Phylo.Newick.Tree:
    tree = Phylo.read(
        f"ClonesModelling/data/patient_data/trees_with_branch_lengths/{patient}.nwk",
        "newick",
    )
    assert isinstance(tree, Phylo.Newick.Tree)
    return tree


def get_patient_branch_lengths(patient: str) -> np.ndarray:
    tree = get_patient_tree(patient)
    return np.array(
        [
            node.branch_length
            for node in tree.find_clades()
            if node.branch_length is not None
        ]
    )


def get_branch_lengths() -> dict[str, np.ndarray]:
    branch_lengths: dict[str, np.ndarray] = {}
    for patient_dot_csv in os.listdir(
        "ClonesModelling/data/patient_data/trees_with_branch_lengths"
    ):
        patient = patient_dot_csv.split(".")[0]
        branch_lengths[patient] = get_patient_branch_lengths(patient)
    return branch_lengths


def get_patient_tree_balance(patient: str) -> float:
    """
    Read or calculate (and record) the tree balance of a patient's tree.
    """
    tree_balances: dict[str, float] = {}
    if os.path.exists("ClonesModelling/data/patient_data/tree_balance.json"):
        with open(
            "ClonesModelling/data/patient_data/tree_balance.json", "r", encoding="utf-8"
        ) as file:
            tree_balances = json.load(file)
        assert isinstance(tree_balances, dict)
        if patient in tree_balances:
            return tree_balances[patient]
    tree_balance = j_one(get_patient_tree(patient))
    tree_balances[patient] = tree_balance
    with open(
        "ClonesModelling/data/patient_data/tree_balance.json", "w", encoding="utf-8"
    ) as file:
        json.dump(tree_balances, file)
    return tree_balance


def get_tree_balances() -> dict[str, float]:
    tree_balances: dict[str, float] = {}
    for patient_dot_csv in os.listdir(
        "ClonesModelling/data/patient_data/trees_with_branch_lengths"
    ):
        patient = patient_dot_csv.split(".")[0]
        tree_balances[patient] = get_patient_tree_balance(patient)
    return tree_balances
