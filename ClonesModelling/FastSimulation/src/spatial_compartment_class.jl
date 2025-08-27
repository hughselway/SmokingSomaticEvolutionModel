module SpatialCompartmentClass

using Random

using ..CompartmentClass
using ..CellLatticeClass
using ..CellPhylogenyClass
using ..SpatialCellPhylogenyClass
using ..MutationPhylogenyClass

struct SpatialCompartment <: CompartmentClass.AbstractCompartment
    name::String
    randomiser::Random.Xoshiro
    mutation_phylogeny::Union{MutationPhylogenyClass.MutationPhylogeny,Nothing}
    cell_phylogeny::Union{
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
        SpatialCellPhylogenyClass.EmptyCellPhylogeny,
    }
    cell_type::CellPhylogenyClass.CellType
    initial_cell_number::Int
    protection_coefficient::Float64
    to_be_counted::Bool
    affected_by_immune_system::Bool
    included_in_competition::Bool
    destination_compartment_name::Union{String,Nothing}
    yearly_counters::Union{CompartmentClass.YearlyCounters,Nothing}
end

function SpatialCompartment(
    randomiser::Random.Xoshiro,
    mutation_phylogeny::Union{MutationPhylogenyClass.MutationPhylogeny,Nothing},
    cell_phylogeny::Union{
        SpatialCellPhylogenyClass.SpatialCellPhylogeny,
        SpatialCellPhylogenyClass.EmptyCellPhylogeny,
    },
    cell_type::CellPhylogenyClass.CellType,
    initial_cell_locations::Vector{Tuple{UInt16,UInt16}},
    cell_lattice::CellLatticeClass.CellLattice,
    protection_coefficient::Float64,
    use_yearly_counters::Bool,
    to_be_counted::Bool,
    affected_by_immune_system::Bool,
    included_in_competition::Bool,
    vary_mutation_rate::Bool,
    compartment_name::String,
    destination_compartment_name::Union{String,Nothing} = nothing,
)::SpatialCompartment
    alive_cell_ids = SpatialCellPhylogenyClass.initialise_cells!(
        cell_phylogeny,
        initial_cell_locations,
        vary_mutation_rate,
    )
    yearly_counters = use_yearly_counters ? CompartmentClass.YearlyCounters() : nothing

    # populate cell lattice with cell ids TODO: move to CellLatticeClass
    if cell_type == CellPhylogenyClass.basal
        for (cell_id, cell_location) in zip(alive_cell_ids, initial_cell_locations)
            cell_lattice[cell_location...] = cell_id
        end
    else
        for (cell_id, cell_location) in zip(alive_cell_ids, initial_cell_locations)
            quiescent_spacing = cell_lattice.quiescent_cell_locations[1][1]
            cell_lattice.quiescent_cell_ids[(cell_location .÷ quiescent_spacing)...] =
                cell_id
        end
    end

    return SpatialCompartment(
        compartment_name,
        randomiser,
        mutation_phylogeny,
        cell_phylogeny,
        cell_type,
        length(alive_cell_ids),
        protection_coefficient,
        to_be_counted,
        affected_by_immune_system,
        included_in_competition,
        destination_compartment_name,
        yearly_counters,
    )
end

function symmetric_divide_cell!(
    spatial_compartment::SpatialCompartment,
    dividing_cell_id::UInt64,
    record_number::UInt16,
    new_cell_location::Tuple{UInt16,UInt16},
)::UInt64
    new_cell_id = SpatialCellPhylogenyClass.symmetric_divide!(
        spatial_compartment.cell_phylogeny,
        dividing_cell_id,
        record_number,
        new_cell_location,
    )
    if spatial_compartment.yearly_counters !== nothing
        spatial_compartment.yearly_counters.this_year_new_cell_count += 1
    end
    return new_cell_id
end

function remove_cell!(
    compartment::SpatialCompartment,
    removed_cell_id::UInt64,
    step::UInt16,
)::Nothing
    CellPhylogenyClass.set_step_of_death_or_removal!(
        compartment.cell_phylogeny,
        removed_cell_id,
        step,
    )
    return nothing
end

function yearly_stats_string(compartment::SpatialCompartment)::String
    return CompartmentClass.yearly_counters_string(compartment)
end

end
