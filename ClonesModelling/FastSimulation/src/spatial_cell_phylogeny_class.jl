module SpatialCellPhylogenyClass

import ..CurrentFitnessClass
import ..CellPhylogenyClass

# so these can be accessed from this namespace if the Empty version is used
import ..CellPhylogenyClass:
    EmptyCellPhylogeny, initialise_cells!, symmetric_divide!, add_mutation!

struct SpatialCell <: CellPhylogenyClass.AbstractCell
    parent_cell_id::Union{UInt64,Nothing}
    step_of_origin::UInt16
    children_cell_ids::Vector{UInt64}
    mutation_rate_multiplier::Union{Ref{Float64},Nothing}
    location::Tuple{UInt16,UInt16}
    step_of_death_or_removal::Ref{Union{UInt16,Nothing}}
end

function get_initial_cell(
    location::Tuple{UInt16,UInt16},
    vary_mutation_rate::Bool,
)::SpatialCell
    return SpatialCell(
        nothing,
        0,
        UInt64[],
        vary_mutation_rate ? 1 : nothing,
        location,
        nothing,
    )
end

mutable struct SpatialCellPhylogeny <: CellPhylogenyClass.AbstractCellPhylogeny
    cell_list::Vector{SpatialCell}
    mutational_profile_ids::Vector{UInt32}
    index::UInt64
    cells_printed_to_disk::UInt64
end

function SpatialCellPhylogeny(total_cell_count_estmate::Int)::SpatialCellPhylogeny
    return SpatialCellPhylogeny(
        Vector{SpatialCell}(undef, total_cell_count_estmate),
        Vector{UInt32}(undef, total_cell_count_estmate),
        1,
        0,
    )
end

function initialise_cells!(
    spatial_cell_phylogeny::SpatialCellPhylogeny,
    locations::Vector{Tuple{UInt16,UInt16}},
    vary_mutation_rate::Bool = false,
)::UnitRange{UInt64}
    # note this assumes it won't immediately overflow the cell list
    for location in locations
        new_cell_id = CellPhylogenyClass.get_new_cell_id(spatial_cell_phylogeny)
        spatial_cell_phylogeny.cell_list[new_cell_id] =
            get_initial_cell(location, vary_mutation_rate)
        spatial_cell_phylogeny.mutational_profile_ids[new_cell_id] = 1
    end
    return spatial_cell_phylogeny.index-length(locations):spatial_cell_phylogeny.index-1
end

function symmetric_divide!(
    spatial_cell_phylogeny::SpatialCellPhylogeny,
    dividing_cell_id::UInt64,
    record_number::UInt16,
    location::Tuple{UInt16,UInt16},
)::UInt64
    dividing_cell = spatial_cell_phylogeny[dividing_cell_id]
    spatial_cell_phylogeny.cell_list[spatial_cell_phylogeny.index] = SpatialCell(
        dividing_cell_id,
        record_number,
        UInt32[],
        if dividing_cell.mutation_rate_multiplier === nothing
            nothing
        else
            dividing_cell.mutation_rate_multiplier[]
        end,
        location,
        nothing,
    )
    spatial_cell_phylogeny.mutational_profile_ids[spatial_cell_phylogeny.index] =
        spatial_cell_phylogeny.mutational_profile_ids[dividing_cell_id]

    child_cell_id =
        CellPhylogenyClass.add_child!(spatial_cell_phylogeny, dividing_cell_id)
    return child_cell_id
end

end
