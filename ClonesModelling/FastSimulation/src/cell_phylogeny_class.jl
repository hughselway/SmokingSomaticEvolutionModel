module CellPhylogenyClass

import Base: getindex, iterate, string

using ..CurrentFitnessClass

@enum CellType begin
    basal
    quiescent
end

@enum DivisionType begin
    symmetric_division
    asymmetric_division
    symmetric_differentiation
end

abstract type AbstractCell end

function is_alive(cell::AbstractCell)::Bool
    return cell.step_of_death_or_removal[] === nothing
end

struct Cell <: AbstractCell
    parent_cell_id::Union{UInt64,Nothing}
    step_of_origin::UInt16
    children_cell_ids::Vector{UInt64}
    current_fitness::CurrentFitnessClass.CurrentFitness
    mutation_rate_multiplier::Union{Ref{Float64},Nothing}
    step_of_death_or_removal::Ref{Union{UInt16,Nothing}}
end

function string(cell::Cell)::String
    return (
        "p$(cell.parent_cell_id),s$(cell.step_of_origin),c$(Int.(cell.children_cell_ids))" *
        (
            if cell.mutation_rate_multiplier === nothing
                ""
            else
                ",m$(cell.mutation_rate_multiplier[])"
            end
        ) *
        (
            if cell.step_of_death_or_removal[] === nothing
                ""
            else
                ",d$(cell.step_of_death_or_removal[])"
            end
        )
    )
end

function get_initial_cell(vary_mutation_rate::Bool)::Cell
    return Cell(
        nothing,
        0,
        UInt64[],
        CurrentFitnessClass.CurrentFitness(0, 0),
        vary_mutation_rate ? 1 : nothing,
        nothing,
    )
end

abstract type AbstractCellPhylogeny end

function iterate(cp::AbstractCellPhylogeny, state = 1)
    state > cp.index - 1 && return nothing
    return (cp.cell_list[state], state + 1)
end

mutable struct CellPhylogeny <: AbstractCellPhylogeny
    cell_list::Vector{Cell}
    mutational_profile_ids::Vector{UInt32}
    index::UInt64
    cells_printed_to_disk::UInt64
end

function CellPhylogeny(total_cell_count_estmate::Int)::CellPhylogeny
    return CellPhylogeny(
        Vector{Cell}(undef, total_cell_count_estmate),
        Vector{UInt32}(undef, total_cell_count_estmate),
        1,
        0,
    )
end

function string(cp::CellPhylogeny)::String
    return "CellPhylogeny:\n\t" * join(
        [
            "$i: " *
            string(cp.cell_list[i]) *
            " -- mid $(cp.mutational_profile_ids[i])" for i in 1:cp.index-1
        ],
        "\n\t",
    )
end

mutable struct EmptyCellPhylogeny <: AbstractCellPhylogeny
    index::UInt64
end

function EmptyCellPhylogeny()::EmptyCellPhylogeny
    return EmptyCellPhylogeny(1)
end

function getindex(cp::AbstractCellPhylogeny, cell_id::UInt64)::AbstractCell
    return cp.cell_list[cell_id-cp.cells_printed_to_disk]
end

function get_new_cell_id(cp::AbstractCellPhylogeny)::UInt64
    cp.index += 1
    if cp.index > length(cp.cell_list)
        println(
            "Resizing cell list from $(length(cp.cell_list)) to ",
            2 * length(cp.cell_list),
        )
        resize!(cp.cell_list, 2 * length(cp.cell_list))
        resize!(cp.mutational_profile_ids, 2 * length(cp.mutational_profile_ids))
    end
    return cp.index + cp.cells_printed_to_disk - 1
end

function initialise_cells!(
    cp::CellPhylogeny,
    initial_cell_count::Int,
    vary_mutation_rate::Bool = false,
)::UnitRange{UInt64}
    for _ in 1:initial_cell_count
        new_cell_id = get_new_cell_id(cp)
        cp.cell_list[new_cell_id] = get_initial_cell(vary_mutation_rate)
        cp.mutational_profile_ids[new_cell_id] = 1
    end
    return cp.index-initial_cell_count:cp.index-1
end

function initialise_cells!(
    cp::EmptyCellPhylogeny,
    initial_cell_count::Int,
    ::Bool = false,
)::UnitRange{UInt64}
    cp.index += initial_cell_count
    return cp.index-initial_cell_count:cp.index-1
end

function initialise_cells!(
    cp::EmptyCellPhylogeny,
    locations::Vector{Tuple{UInt16,UInt16}},
    ::Bool = false,
)::UnitRange{UInt64}
    return initialise_cells!(cp, length(locations))
end

function symmetric_divide!(
    cell_phylogeny::CellPhylogeny,
    dividing_cell_id::UInt64,
    step::UInt16,
)::UInt64
    dividing_cell = cell_phylogeny.cell_list[dividing_cell_id]
    cell_phylogeny.cell_list[cell_phylogeny.index] = Cell(
        dividing_cell_id,
        step,
        UInt64[],
        CurrentFitnessClass.copy(dividing_cell.current_fitness),
        if dividing_cell.mutation_rate_multiplier === nothing
            nothing
        else
            dividing_cell.mutation_rate_multiplier[]
        end,
        nothing,
    )
    cell_phylogeny.mutational_profile_ids[cell_phylogeny.index] =
        cell_phylogeny.mutational_profile_ids[dividing_cell_id]
    return add_child!(cell_phylogeny, dividing_cell_id)
end

function add_child!(
    cell_phylogeny::AbstractCellPhylogeny,
    parent_cell_id::UInt64,
)::UInt64
    child_cell_id = get_new_cell_id(cell_phylogeny)
    if cell_phylogeny.cell_list[parent_cell_id].children_cell_ids !== nothing
        push!(cell_phylogeny.cell_list[parent_cell_id].children_cell_ids, child_cell_id)
    else
        cell_phylogeny.cell_list[parent_cell_id].children_cell_ids = [child_cell_id]
    end
    return child_cell_id
end

function symmetric_divide!(
    cell_phylogeny::EmptyCellPhylogeny,
    ::UInt64,
    ::UInt16,
    ::Tuple{UInt16,UInt16} = nothing,
)::UInt64
    cell_phylogeny.index += 1
    return cell_phylogeny.index - 1
end

function count_divisions(
    cp::AbstractCellPhylogeny,
    cell_id::UInt64,
    subtree_root_id::Union{UInt64,Nothing} = nothing,
)::UInt32
    divisions = length(cp.cell_list[cell_id].children_cell_ids)
    current_cell_id = cp.cell_list[cell_id].parent_cell_id
    previous_cell_id = cell_id
    while current_cell_id !== nothing &&
        (subtree_root_id === nothing || current_cell_id != subtree_root_id)
        divisions += findfirst(
            cp.cell_list[current_cell_id].children_cell_ids .== previous_cell_id,
        )
        previous_cell_id = current_cell_id
        current_cell_id = cp.cell_list[current_cell_id].parent_cell_id
    end
    return divisions
end

function add_mutation!(
    cell_phylogeny::AbstractCellPhylogeny,
    cell_id::UInt64,
    new_mutational_profile_id::UInt32,
    fitness_change::Float64,
    smoking_driver_fitness_augmentation::Float64,
    protection_coefficient::Float64,
    mutation_rate_multiplier_change::Float64,
)::Nothing
    cell_phylogeny.mutational_profile_ids[cell_id] = new_mutational_profile_id
    if typeof(cell_phylogeny) <: CellPhylogeny
        # otherwise spatial, and fitness is stored in lattice
        CurrentFitnessClass.apply_fitness_change!(
            cell_phylogeny[cell_id].current_fitness,
            fitness_change,
            smoking_driver_fitness_augmentation,
            protection_coefficient,
        )
    end
    if mutation_rate_multiplier_change != 0
        cell_phylogeny[cell_id].mutation_rate_multiplier[] +=
            mutation_rate_multiplier_change
    end
    return nothing
end

function add_mutation!(
    ::EmptyCellPhylogeny,
    ::UInt64,
    ::UInt32,
    ::Float64,
    ::Float64,
    ::Float64,
    ::Float64,
)::Nothing
    return nothing
end

cell_count(cp::CellPhylogeny) = cp.cells_printed_to_disk + cp.index - 1

function alive_cell_count(cp::CellPhylogeny)::UInt32
    alive_cell_count = 0
    for cell in cp.cell_list[1:cp.index-1]
        if cell.step_of_death_or_removal === nothing
            alive_cell_count += 1
        end
    end
    return alive_cell_count
end

function set_step_of_death_or_removal!(
    cell_phylogeny::AbstractCellPhylogeny,
    cell_id::UInt64,
    step::UInt16,
)::Nothing
    cell_phylogeny.cell_list[cell_id].step_of_death_or_removal[] = step
    return nothing
end

function set_step_of_death_or_removal!(
    ::EmptyCellPhylogeny,
    ::UInt64,
    ::UInt16,
)::Nothing
    return nothing
end

function get_alive_cell_ids(
    cell_phylogeny::AbstractCellPhylogeny,
    subtree_root_id::UInt64,
)::Vector{UInt64}
    alive_cell_ids = Vector{UInt64}(undef, 2^10)
    alive_cell_index = 1
    stack = Vector{UInt64}(undef, 2^10)
    stack[1] = subtree_root_id
    stack_index = 2
    while stack_index > 1
        current_cell_id = stack[stack_index-1]
        stack_index -= 1
        if is_alive(cell_phylogeny[current_cell_id])
            if alive_cell_index > length(alive_cell_ids)
                resize!(alive_cell_ids, 2 * length(alive_cell_ids))
            end
            alive_cell_ids[alive_cell_index] = current_cell_id
            alive_cell_index += 1
        end
        for child_id in cell_phylogeny[current_cell_id].children_cell_ids
            if stack_index > length(stack)
                resize!(stack, 2 * length(stack))
            end
            stack[stack_index] = child_id
            stack_index += 1
        end
    end
    return alive_cell_ids[1:alive_cell_index-1]
end

function get_root_cell_ids(cell_phylogeny::AbstractCellPhylogeny)::Vector{UInt64}
    root_cell_ids = Vector{UInt64}()
    for (index, cell) in enumerate(cell_phylogeny)
        if cell.parent_cell_id === nothing
            push!(root_cell_ids, index + cell_phylogeny.cells_printed_to_disk)
        end
    end
    return root_cell_ids
end

end
