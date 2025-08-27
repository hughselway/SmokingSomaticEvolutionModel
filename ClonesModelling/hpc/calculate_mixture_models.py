import os
import sys

from ..id_test.read_data import IdentifiabilityDataset


def calculate_mixture_models():
    ## reads in each simulation via the IdentifiabilityDataset, which calculates all MMs
    idt_id = sys.argv[1]
    id = IdentifiabilityDataset(
        "/cluster/project2/clones_modelling/identifiability_test", idt_id
    )
    failed_replicates = []
    for replicate_index in id.replicate_indexer:
        try:
            id.get_replicate(replicate_index)
        except ValueError:
            failed_replicates.append(replicate_index)
    if failed_replicates:
        print(
            f"Failed to calculate mixture models for replicates {failed_replicates}; "
            f"succeeded for {[x for x in id.replicate_indexer if x not in failed_replicates]}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 check_pairwise_distances.py <idt_id>; using default for now"
        )
        assert len(sys.argv) == 1
        sys.argv.append("idt_2025-02-05_14-53-34")
    calculate_mixture_models()
