module CondensedPhylogenyClass

import Base: getproperty, string

using ..CellPhylogenyClass
using ..MutationCountClass
using ..MutationPhylogenyClass
using ..SpatialCellPhylogenyClass

struct Cell
    id::UInt64
    step_of_origin::UInt16
    mutation_rate_multiplier::Union{Ref{Float64},Nothing}
    is_alive::Bool
end

struct Node
    parent_node_index::Union{UInt64,Nothing}
    children_node_indices::Vector{UInt64}
    divisions_on_branch::UInt32
    mutations_on_branch::UInt32
    cell::Union{Cell,Nothing}
end

function Node(
    parent_node_index::Union{UInt64,Nothing},
    children_node_indices::Vector{UInt64},
    divisions_on_branch::Integer,
    mutations_on_branch::Integer,
    cell_id::UInt64,
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
)::Node
    return Node(
        parent_node_index,
        children_node_indices,
        UInt32(divisions_on_branch),
        UInt32(mutations_on_branch),
        Cell(
            cell_id,
            cell_phylogeny[cell_id].step_of_origin,
            cell_phylogeny[cell_id].mutation_rate_multiplier,
            is_alive(cell_id, cell_phylogeny, subset_cell_ids),
        ),
    )
end

function children_node_indices_string(children_node_indices::Vector{UInt64})::String
    if (length(children_node_indices) == 0)
        return "-"
    end
    return join(map(x -> x, children_node_indices), "-")
end

function string(node::Node)::String
    return (
        "p$(node.parent_node_index),c" *
        children_node_indices_string(node.children_node_indices) *
        "," *
        "m$(node.mutations_on_branch),d$(node.divisions_on_branch),cid" *
        (
            if node.cell === nothing
                "-"
            else
                "$(node.cell.id)" * (node.cell.is_alive ? "" : "!")
            end
        )
    )
end

function Node(
    parent_node_index::Union{UInt64,Nothing},
    children_node_indices::Vector{UInt64},
    divisions_on_branch::Integer,
    mutations_on_branch::Integer,
    ::Nothing,
    ::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    ::Union{Nothing,Vector{UInt64}},
)::Node
    return Node(
        parent_node_index,
        children_node_indices,
        divisions_on_branch,
        mutations_on_branch,
        nothing,
    )
end

function add_parent(
    previous_root::Node,
    new_parent_index::Union{UInt64,Nothing},
    divisions_on_branch::Integer,
    mutations_on_branch::Integer,
)::Node
    new_node = Node(
        new_parent_index,
        previous_root.children_node_indices,
        UInt32(divisions_on_branch) + previous_root.divisions_on_branch,
        UInt32(mutations_on_branch) + previous_root.mutations_on_branch,
        previous_root.cell,
    )
    return new_node
end
function add_parent(
    node::Node,
    new_parent_index::Integer,
    divisions_on_branch::Integer,
    mutations_on_branch::Integer,
)::Node
    return add_parent(
        node,
        UInt64(new_parent_index),
        divisions_on_branch,
        mutations_on_branch,
    )
end

struct CondensedPhylogeny
    nodes::Vector{Node}
    root_mutational_profile_id::Union{UInt32,Nothing}
end

function alive_cell_count(tree::CondensedPhylogeny)::Int
    return sum(node.cell !== nothing && node.cell.is_alive for node in tree.nodes)
end

function trace_most_recent_common_ancestor(
    parent_profile_id::UInt32,
    profile_id::UInt32,
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    verbose::Bool = false,
)::Tuple{UInt32,MutationCountClass.MutationCount,MutationCountClass.MutationCount}
    parent_mutation_count = MutationCountClass.MutationCount()
    child_mutation_count = MutationCountClass.MutationCount()
    while parent_profile_id != profile_id
        while parent_profile_id > profile_id
            parent_mutation_count = MutationCountClass.copy_incremented(
                parent_mutation_count,
                mutation_phylogeny[parent_profile_id].mutation_fitness >= 0,
                mutation_phylogeny[parent_profile_id].smoking_signature,
            )
            parent_profile_id = mutation_phylogeny[parent_profile_id].parent_profile_id
            if verbose
                print("p", parent_profile_id, " ")
            end
        end
        while profile_id > parent_profile_id
            child_mutation_count = MutationCountClass.copy_incremented(
                child_mutation_count,
                mutation_phylogeny[profile_id].mutation_fitness >= 0,
                mutation_phylogeny[profile_id].smoking_signature,
            )
            profile_id = mutation_phylogeny[profile_id].parent_profile_id
            if verbose
                print("c", profile_id, " ")
            end
        end
        if verbose
            println(
                " -- p:",
                MutationCountClass.total_mutations(parent_mutation_count),
                " c:",
                MutationCountClass.total_mutations(child_mutation_count),
            )
        end
    end
    return profile_id, parent_mutation_count, child_mutation_count
end

function mutations_on_branch(
    parent_profile_id::Union{UInt32,Nothing},
    profile_id::UInt32,
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    verbose::Bool = false,
)::UInt32
    if parent_profile_id === nothing
        return MutationPhylogenyClass.mutation_count(mutation_phylogeny, profile_id)
    end

    _, _, mutation_count_on_branch = trace_most_recent_common_ancestor(
        parent_profile_id,
        profile_id,
        mutation_phylogeny,
        verbose,
    )
    total_mutations = MutationCountClass.total_mutations(mutation_count_on_branch)
    return total_mutations
end

function record_condensed_phylogeny(
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
)::String
    condensed_phylogeny =
        condense_phylogeny(cell_phylogeny, mutation_phylogeny, subset_cell_ids)
    assert_valid(condensed_phylogeny, cell_phylogeny, mutation_phylogeny)
    return get_nwk_string(condensed_phylogeny, true)
end

function condense_phylogeny(
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
)::CondensedPhylogeny
    non_empty_subtrees = Vector{CondensedPhylogeny}()
    for root_cell_id in CellPhylogenyClass.get_root_cell_ids(cell_phylogeny)
        this_root_subtree = condense_phylogeny(
            cell_phylogeny,
            mutation_phylogeny,
            subset_cell_ids,
            root_cell_id,
        )
        if length(this_root_subtree.nodes) > 0
            push!(non_empty_subtrees, this_root_subtree)
        end
    end
    if length(non_empty_subtrees) == 1
        full_tree = CondensedPhylogeny(non_empty_subtrees[1].nodes, nothing)
        full_tree.nodes[end] = add_parent(
            full_tree.nodes[end],
            nothing,
            0,
            MutationPhylogenyClass.mutation_count(
                mutation_phylogeny,
                non_empty_subtrees[1].root_mutational_profile_id,
            ),
        )
    else
        full_tree = merge_trees(
            non_empty_subtrees,
            nothing,
            cell_phylogeny,
            mutation_phylogeny,
            subset_cell_ids,
        )
    end
    return full_tree
end

function condense_phylogeny(
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
    subtree_root_id::UInt64,
)::CondensedPhylogeny
    alive_cell_ids =
        CellPhylogenyClass.get_alive_cell_ids(cell_phylogeny, subtree_root_id)
    if subset_cell_ids !== nothing
        alive_cell_ids =
            [cell_id for cell_id in alive_cell_ids if cell_id in subset_cell_ids]
    end
    if length(alive_cell_ids) == 0
        return CondensedPhylogeny(Node[], nothing)
    end
    current_root_id = alive_cell_ids[end] # a childless leaf
    current_subtree = CondensedPhylogeny(
        [
            Node(
                nothing,
                UInt64[],
                0,
                0,
                current_root_id,
                cell_phylogeny,
                subset_cell_ids,
            ),
        ],
        cell_phylogeny.mutational_profile_ids[current_root_id],
    )
    while alive_cell_count(current_subtree) < length(alive_cell_ids)
        @assert subtree_root_id === nothing || current_root_id != subtree_root_id (
            "Alive cells not found before root"
        )
        current_subtree, current_root_id = parse_parent_subtree(
            cell_phylogeny,
            mutation_phylogeny,
            current_subtree,
            current_root_id,
            subset_cell_ids,
        )
    end
    return current_subtree
end

function string(condensed_phylogeny::CondensedPhylogeny)::String
    return (
        "CondensedPhylogeny:\n\t" *
        join(
            [
                "$i: " * string(node) for
                (i, node) in enumerate(condensed_phylogeny.nodes)
            ],
            "\n\t",
        ) *
        "\n\troot mid: $(condensed_phylogeny.root_mutational_profile_id)"
    )
end

function parse_parent_subtree(
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    subtree::CondensedPhylogeny,
    subtree_root_id::UInt64,
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
)::Tuple{CondensedPhylogeny,UInt64}
    # go up through parents until there's a branch point (multiple children) or an
    # alive cell, then add the new subbranch(es) onto the subtree and return it
    current_root_id = cell_phylogeny[subtree_root_id].parent_cell_id
    @assert current_root_id !== nothing "No parent for $(subtree_root_id)"
    previous_root_id = subtree_root_id
    divisions_on_branch = 1
    while length(cell_phylogeny[current_root_id].children_cell_ids) == 1
        if is_alive(current_root_id, cell_phylogeny, subset_cell_ids)
            return (
                add_trivial_branch_to_subtree(
                    cell_phylogeny,
                    mutation_phylogeny,
                    current_root_id,
                    subtree,
                    divisions_on_branch,
                    subset_cell_ids,
                ),
                current_root_id,
            )
        end
        previous_root_id = current_root_id
        current_root_id = cell_phylogeny[current_root_id].parent_cell_id
        divisions_on_branch += findfirst(
            cell_phylogeny.cell_list[current_root_id].children_cell_ids .==
            previous_root_id,
        )
    end

    child_subtrees = Vector{CondensedPhylogeny}([subtree])
    for child_id in cell_phylogeny[current_root_id].children_cell_ids
        if child_id == previous_root_id
            continue
        end
        this_child_subtree = condense_phylogeny(
            cell_phylogeny,
            mutation_phylogeny,
            subset_cell_ids,
            child_id,
        )
        if length(this_child_subtree.nodes) == 0
            continue
        end
        push!(child_subtrees, this_child_subtree)
    end

    if length(child_subtrees) == 1 &&
       !is_alive(current_root_id, cell_phylogeny, subset_cell_ids)
        # then this doesn't need to be a node in the phylogeny; just keep going up 
        # until the next branch
        return parse_parent_subtree(
            cell_phylogeny,
            mutation_phylogeny,
            subtree,
            current_root_id,
            subset_cell_ids,
        )
    end
    return (
        merge_trees(
            child_subtrees,
            current_root_id,
            cell_phylogeny,
            mutation_phylogeny,
            subset_cell_ids,
        ),
        current_root_id,
    )
end

function merge_trees(
    non_empty_subtrees::Vector{CondensedPhylogeny},
    common_parent_id::Union{UInt64,Nothing},
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
)::CondensedPhylogeny
    ## add new cell as a common parent of all the subtrees (split across multiple
    ## internal nodes due to mutations between the splitting points)
    subtree_root_total_mutations = [
        MutationPhylogenyClass.mutation_count(
            mutation_phylogeny,
            subtree.root_mutational_profile_id,
        ) for subtree in non_empty_subtrees
    ]
    mutations_on_child_branches = (
        if common_parent_id === nothing
            subtree_root_total_mutations
        else
            [
                mutations_on_branch(
                    cell_phylogeny.mutational_profile_ids[common_parent_id],
                    subtree.root_mutational_profile_id,
                    mutation_phylogeny,
                ) for subtree in non_empty_subtrees
            ]
        end
    )
    mutations_at_split_from_common_parent =
        subtree_root_total_mutations .- mutations_on_child_branches
    # reorder all the subtrees by the number of mutations at the split, largest first 
    reordered_indices = sortperm(mutations_at_split_from_common_parent, rev = true)
    non_empty_subtrees = non_empty_subtrees[reordered_indices]
    mutations_on_child_branches = mutations_on_child_branches[reordered_indices]
    subtree_root_total_mutations = subtree_root_total_mutations[reordered_indices]
    mutations_at_split_from_common_parent =
        mutations_at_split_from_common_parent[reordered_indices]
    @assert issorted(mutations_at_split_from_common_parent, rev = true)

    merged_tree_nodes = Vector{Node}()
    for (i, tree) in enumerate(non_empty_subtrees)
        @assert length(tree.nodes) > 0
        # append this tree's nodes to the merged tree, but with all indices incremented
        index_increment = length(merged_tree_nodes)
        for node in tree.nodes
            push!(merged_tree_nodes, increment_indices(node, index_increment))
        end
        if i == 1 &&
           common_parent_id !== nothing &&
           is_alive(common_parent_id, cell_phylogeny, subset_cell_ids)
            # add new internal node, with the "new root" and the first subtree as children
            # add mutations_on_child_branches[1] as the mutations on the branch to the subtree
            # add total_mutations_of_root - mutations_at_split_from_new_root[1] as the mutations on the branch to the new root
            this_subtree_root_index::UInt64 = length(merged_tree_nodes)
            merged_tree_nodes[end] = add_parent(
                merged_tree_nodes[end],
                this_subtree_root_index + 2,
                1,
                mutations_on_child_branches[1],
            )
            @assert MutationPhylogenyClass.mutation_count(
                mutation_phylogeny,
                cell_phylogeny.mutational_profile_ids[common_parent_id],
            ) >= mutations_at_split_from_common_parent[1] (
                "common parent $(common_parent_id) has " *
                "$(MutationPhylogenyClass.mutation_count(
                    mutation_phylogeny,
                    cell_phylogeny.mutational_profile_ids[common_parent_id],
                )) mutations, " *
                "subtree has $(mutations_at_split_from_common_parent[1])"
            )
            push!(
                merged_tree_nodes,
                Node( # "new root" node, actually a leaf
                    this_subtree_root_index + 2,
                    Vector{UInt64}(),
                    0,
                    MutationPhylogenyClass.mutation_count(
                        mutation_phylogeny,
                        cell_phylogeny.mutational_profile_ids[common_parent_id],
                    ) - mutations_at_split_from_common_parent[1],
                    common_parent_id,
                    cell_phylogeny,
                    subset_cell_ids,
                ),
                Node( # internal node
                    if length(non_empty_subtrees) > i
                        UInt64(
                            this_subtree_root_index +
                            length(non_empty_subtrees[i+1].nodes) +
                            3,
                        )
                    else
                        nothing
                    end,
                    UInt64.([this_subtree_root_index, this_subtree_root_index + 1]),
                    0,
                    if length(non_empty_subtrees) > 1
                        mutations_at_split_from_common_parent[1] -
                        mutations_at_split_from_common_parent[2]
                    else
                        0
                    end,
                    nothing,
                ),
            )
        elseif i == 1
            # then the "new root" is not alive, so we don't need to add an extra leaf
            @assert length(non_empty_subtrees) > 1
            merged_tree_nodes[end] = add_parent(
                merged_tree_nodes[end],
                length(merged_tree_nodes) + (
                    if (length(non_empty_subtrees) > 1)
                        (length(non_empty_subtrees[2].nodes) + 1)
                    else
                        2
                    end
                ),
                1,
                subtree_root_total_mutations[1] -
                mutations_at_split_from_common_parent[2],
            )
        else
            # add this subtree's root as a child of the new root
            merged_tree_nodes[end] = add_parent(
                merged_tree_nodes[end],
                length(merged_tree_nodes) + 1,
                1,
                mutations_on_child_branches[i],
            )
            # add an internal node
            push!(
                merged_tree_nodes,
                Node(
                    if length(non_empty_subtrees) > i
                        UInt64(
                            length(merged_tree_nodes) +
                            length(non_empty_subtrees[i+1].nodes) +
                            2,
                        )
                    else
                        nothing
                    end,
                    UInt64.([
                        length(merged_tree_nodes) - length(tree.nodes),
                        length(merged_tree_nodes),
                    ]),
                    0,
                    if length(non_empty_subtrees) > i
                        mutations_at_split_from_common_parent[i] -
                        mutations_at_split_from_common_parent[i+1]
                    else
                        0
                    end,
                    nothing,
                ),
            )
        end
        assert_root_children_match(merged_tree_nodes)
    end
    if common_parent_id !== nothing
        return CondensedPhylogeny(
            merged_tree_nodes,
            trace_most_recent_common_ancestor(
                cell_phylogeny.mutational_profile_ids[common_parent_id],
                non_empty_subtrees[end].root_mutational_profile_id,
                mutation_phylogeny,
            )[1],
        )
    end
    return CondensedPhylogeny(merged_tree_nodes, nothing)
end

function assert_root_children_match(nodes::Vector{Node}, verbose::Bool = false)::Nothing
    if verbose
        println(
            "children node indices: " *
            join(
                map(x -> children_node_indices_string(x.children_node_indices), nodes),
                ",",
            ) *
            "\nparent node indices: " *
            join(map(x -> x.parent_node_index, nodes), ","),
        )
    end
    for child_node_index in nodes[end].children_node_indices
        @assert nodes[child_node_index].parent_node_index == length(nodes) (
            "node $child_node_index, $(nodes[child_node_index]) should have parent " *
            "$(length(nodes))"
        )
    end
end

function increment_indices(node::Node, increment::Integer)::Node
    return Node(
        if node.parent_node_index === nothing
            nothing
        else
            increment + node.parent_node_index
        end,
        map(x -> increment + x, node.children_node_indices),
        node.divisions_on_branch,
        node.mutations_on_branch,
        node.cell,
    )
end

function add_trivial_branch_to_subtree(
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    new_cell_id::UInt64,
    subtree::CondensedPhylogeny,
    divisions_on_branch::Integer,
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
)::CondensedPhylogeny
    # trivial in the sense that there's no branching points between the previous tip
    # and the new tip
    @assert length(subtree.nodes) > 0 "No previous tip to add branch to"
    @assert CellPhylogenyClass.is_alive(cell_phylogeny[new_cell_id]) "New cell is dead"
    previous_root_index::UInt64 = length(subtree.nodes)
    new_leaf_index::UInt64 = previous_root_index + 1
    new_internal_root_index::UInt64 = previous_root_index + 2

    mrca, _, mutation_count_on_branch = trace_most_recent_common_ancestor(
        cell_phylogeny.mutational_profile_ids[new_cell_id],
        subtree.root_mutational_profile_id,
        mutation_phylogeny,
    )
    mutations_on_branch = MutationCountClass.total_mutations(mutation_count_on_branch)

    new_leaf = Node(
        new_internal_root_index,
        UInt64[],
        0,
        MutationPhylogenyClass.mutation_count(
            mutation_phylogeny,
            cell_phylogeny.mutational_profile_ids[new_cell_id],
        ) - (
            MutationPhylogenyClass.mutation_count(
                mutation_phylogeny,
                subtree.root_mutational_profile_id,
            ) - mutations_on_branch
        ),
        new_cell_id,
        cell_phylogeny,
        subset_cell_ids,
    )
    new_internal_root =
        Node(nothing, UInt64.([previous_root_index, new_leaf_index]), 0, 0, nothing)
    new_subtree = CondensedPhylogeny(
        vcat(
            subtree.nodes[1:end-1],
            [
                add_parent(
                    subtree.nodes[end],
                    length(subtree.nodes) + 2,
                    divisions_on_branch,
                    mutations_on_branch,
                ),
                new_leaf,
                new_internal_root,
            ],
        ),
        mrca,
    )
    assert_root_children_match(new_subtree.nodes)
    return new_subtree
end

function assert_valid(
    condensed_phylogeny::CondensedPhylogeny,
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
    },
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
)::Nothing
    for (i, node) in enumerate(condensed_phylogeny.nodes)
        # each child should have it as a parent; its parent should have it as a child
        for child_index in node.children_node_indices
            @assert child_index < i
            @assert condensed_phylogeny.nodes[child_index].parent_node_index == i
        end
        if node.parent_node_index !== nothing
            @assert node.parent_node_index > i
            @assert (
                i in
                condensed_phylogeny.nodes[node.parent_node_index].children_node_indices
            ) (
                "Node $(i), $(node.cell) has parent $(node.parent_node_index), " *
                "$(condensed_phylogeny.nodes[node.parent_node_index].children_node_indices)"
            )
        end
    end
    # assert no two nodes have the same cell
    cell_ids = Set{UInt64}()
    for node in condensed_phylogeny.nodes
        if node.cell === nothing
            continue
        end
        @assert !(node.cell.id in cell_ids) (
            "$(node.cell.id) already in $(Int.(cell_ids)), " *
            "children are $(Int.(node.children_node_indices))" *
            "parent chain $(parent_chain(condensed_phylogeny, node.parent_node_index))"
        )
        push!(cell_ids, node.cell.id)
    end

    # assert each leaf node has a cell and has consistent mutation counts
    for node in condensed_phylogeny.nodes
        if length(node.children_node_indices) == 0
            @assert node.cell !== nothing
            @assert node.cell.is_alive
            total_mutations_from_condensed_phylogeny = node.mutations_on_branch
            parent_index = node.parent_node_index
            while parent_index !== nothing
                total_mutations_from_condensed_phylogeny +=
                    condensed_phylogeny.nodes[parent_index].mutations_on_branch
                parent_index = condensed_phylogeny.nodes[parent_index].parent_node_index
            end
            @assert total_mutations_from_condensed_phylogeny ==
                    MutationPhylogenyClass.mutation_count(
                mutation_phylogeny,
                cell_phylogeny.mutational_profile_ids[node.cell.id],
            ) (
                "Total mutations from condensed phylogeny: " *
                "$(total_mutations_from_condensed_phylogeny), " *
                "mutation count: " *
                "$(MutationPhylogenyClass.mutation_count(
                    mutation_phylogeny,
                    cell_phylogeny.mutational_profile_ids[node.cell.id],
                )) " *
                "root mutations: " *
                "$(MutationPhylogenyClass.mutation_count(
                    mutation_phylogeny,
                    condensed_phylogeny.root_mutational_profile_id,
                ))"
            )
        end
    end
end

function get_nwk_string(
    condensed_phylogeny::CondensedPhylogeny,
    mutations_as_branch_lengths::Bool,
)::String
    if length(condensed_phylogeny.nodes) == 0
        return ""
    end
    io = IOBuffer()
    write_node_to_nwk(
        condensed_phylogeny,
        io,
        mutations_as_branch_lengths,
        length(condensed_phylogeny.nodes),
    )
    print(io, ";")
    return String(take!(io))
end

function write_node_to_nwk(
    condensed_phylogeny::CondensedPhylogeny,
    io::IO,
    mutations_as_branch_lengths::Bool,
    node_index::Integer,
)::Nothing
    node = condensed_phylogeny.nodes[node_index]
    leaf = length(node.children_node_indices) == 0
    if !leaf
        print(io, "(")
        for child_index in node.children_node_indices
            write_node_to_nwk(
                condensed_phylogeny,
                io,
                mutations_as_branch_lengths,
                child_index,
            )
            if child_index != node.children_node_indices[end]
                print(io, ",")
            end
        end
        print(io, ")")
    end
    if node.cell !== nothing
        print(io, node.cell.id)
        if !node.cell.is_alive
            print(io, "!")
        end
    end
    print(io, ":")
    if mutations_as_branch_lengths
        print(io, node.mutations_on_branch)
    else
        print(io, node.divisions_on_branch)
    end
    return nothing
end

function get_branch_lengths(
    condensed_phylogeny::CondensedPhylogeny,
    mutations_as_branch_lengths::Bool = true,
)::Vector{UInt32}
    if mutations_as_branch_lengths
        return map(node -> node.mutations_on_branch, condensed_phylogeny.nodes)
    else
        return map(node -> node.divisions_on_branch, condensed_phylogeny.nodes)
    end
end

## printing utilities
function nodes_string(condensed_phylogeny::CondensedPhylogeny)::String
    return join(map(node -> node.cell.id, condensed_phylogeny.nodes), ",")
end

function parent_chain(
    condensed_phylogeny::CondensedPhylogeny,
    node_index::Integer,
)::String
    node = condensed_phylogeny.nodes[node_index]
    node_description =
        node.cell.id *
        " ($(node.divisions_on_branch)m; ch: " *
        join(
            map(x -> condensed_phylogeny.nodes[x].cell.id, node.children_node_indices),
            ",",
        ) *
        ")"
    if node.parent_node_index === nothing
        return node_description
    end
    return "$(node_description),$(parent_chain(condensed_phylogeny, node.parent_node_index))"
end

function parent_chain(
    cell_phylogeny::CellPhylogenyClass.AbstractCellPhylogeny,
    cell_id::UInt64,
)::String
    mutational_profile_id = cell_phylogeny.mutational_profile_ids[cell_id]
    parent_chain = "$(cell_id) ($(mutational_profile_id))"
    current_cell_id = cell_phylogeny[cell_id].parent_cell_id
    while current_cell_id !== nothing
        mutational_profile_id = cell_phylogeny.mutational_profile_ids[current_cell_id]
        parent_chain *= " -> $(current_cell_id) ($(mutational_profile_id))"
        current_cell_id = cell_phylogeny[current_cell_id].parent_cell_id
    end
    return parent_chain
end

# and also mutation phylogeny
function parent_chain(
    mutation_phylogeny::MutationPhylogenyClass.MutationPhylogeny,
    profile_id::UInt32,
)::String
    parent_chain = "$(profile_id)"
    current_profile_id = mutation_phylogeny[profile_id].parent_profile_id
    while current_profile_id !== nothing
        parent_chain *= " -> $(current_profile_id)"
        current_profile_id = mutation_phylogeny[current_profile_id].parent_profile_id
    end
    return parent_chain
end

function is_alive(
    cell_id::UInt64,
    cell_phylogeny::CellPhylogenyClass.AbstractCellPhylogeny,
    subset_cell_ids::Union{Nothing,Vector{UInt64}},
)::Bool
    return CellPhylogenyClass.is_alive(cell_phylogeny[cell_id]) &&
           cell_id in subset_cell_ids
end

end
