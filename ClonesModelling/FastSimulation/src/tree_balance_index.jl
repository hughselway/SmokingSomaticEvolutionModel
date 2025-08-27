module TreeBalanceIndex

using ..CondensedPhylogenyClass

function j_one(condensed_phylogeny::CondensedPhylogenyClass.CondensedPhylogeny)::Float64
    # Calculate the tree balance index of a condensed phylogeny, using the index defined
    # in Lemant et al 2022 (https://doi.org/10.1093/sysbio/syac027)
    @assert condensed_phylogeny.root_mutational_profile_id === nothing
    subtree_cell_counts = get_subtree_cell_counts(condensed_phylogeny)

    numerator, denominator = 0, 0
    for (node, subtree_cell_count) in
        zip(condensed_phylogeny.nodes, subtree_cell_counts)
        if node.cell !== nothing || subtree_cell_count == 0
            continue
        end
        denominator += subtree_cell_count
        for child_node_index in node.children_node_indices
            if subtree_cell_counts[child_node_index] == 0
                continue
            end
            numerator +=
                subtree_cell_counts[child_node_index] *
                log2(subtree_cell_counts[child_node_index] / subtree_cell_count)
        end
    end
    return -numerator / denominator
end

function get_subtree_cell_counts(
    condensed_phylogeny::CondensedPhylogenyClass.CondensedPhylogeny,
)::Vector{Int}
    # Count the number of cells in each subtree of a condensed phylogeny
    cell_counts = zeros(Int, length(condensed_phylogeny.nodes))

    for node in condensed_phylogeny.nodes
        if node.cell === nothing
            @assert length(node.children_node_indices) == 2
            continue
        end
        # this is a cell: add one to the cell count of every ancestor
        @assert length(node.children_node_indices) == 0
        parent_node_index = node.parent_node_index
        while parent_node_index !== nothing
            cell_counts[parent_node_index] += 1
            parent_node_index =
                condensed_phylogeny.nodes[parent_node_index].parent_node_index
        end
    end
    @assert (
        cell_counts[end] ==
        CondensedPhylogenyClass.alive_cell_count(condensed_phylogeny)
    ) "cell_counts[end] = $(cell_counts[end]), alive_cell_count = $(CondensedPhylogenyClass.alive_cell_count(condensed_phylogeny))"
    return cell_counts
end

end # module
