module SpatialCellPopulationClass

using OrderedCollections: OrderedDict
using Random
using StatsBase
import Base: string

using ..CompartmentClass
using ..CellLatticeClass
using ..CellPhylogenyClass
using ..CellPopulationClass
using ..CurrentFitnessClass
using ..MutationCountClass
using ..MutationPhylogenyClass
using ..SmokingAffectedParameterClass
using ..SpatialCompartmentClass
using ..SpatialCellPhylogenyClass
using ..Records

@enum EventType begin
    symmetric_differentiation
    asymmetric_division
    immune_death
    ambient_quiescent_division
end

struct SpatialCellPopulation <: CellPopulationClass.AbstractCellPopulation
    randomiser::Random.Xoshiro
    protection_coefficient::Float64
    quiescent_protection_coefficient::Union{Float64,Nothing}
    quiescent_gland_cell_count::Int
    # each cell's fitness is actually the fitness stored in the lattice minus this:
    current_normalisation_constant::Ref{Float64}
    total_mutation_count::Ref{Int}
    cell_lattice::CellLatticeClass.CellLattice
    cell_compartments_dict::OrderedDict{
        String,
        SpatialCompartmentClass.SpatialCompartment,
    }
end

function SpatialCellPopulation(
    randomiser::Random.Xoshiro,
    grid_side_length::Int,
    quiescent_spacing::Int,
    protected_spacing::Int,
    protected_region_radius::Int,
    quiescent_gland_cell_count::Int,
    protection_coefficient::Float64,
    quiescent_protection_coefficient::Union{Float64,Nothing},
    use_yearly_counters::Bool,
    smoking_driver_fitness_augmentation::Float64,
    vary_mutation_rate::Bool,
    record_phylogenies::Bool,
    expected_total_mutational_profiles::Int,
    expected_total_cells::Int,
)::SpatialCellPopulation
    if record_phylogenies
        mutation_phylogeny =
            MutationPhylogenyClass.MutationPhylogeny(expected_total_mutational_profiles)
        cell_phylogeny =
            SpatialCellPhylogenyClass.SpatialCellPhylogeny(expected_total_cells)
    else
        mutation_phylogeny = nothing
        cell_phylogeny = SpatialCellPhylogenyClass.EmptyCellPhylogeny()
    end

    cell_lattice_ids = zeros(UInt64, grid_side_length, grid_side_length)
    quiescent_cell_locations =
        get_quiescent_cell_locations(grid_side_length, quiescent_spacing)
    protected_cell_locations = get_protected_cell_locations(
        grid_side_length,
        protected_spacing,
        protected_region_radius,
    )
    main_compartment_cell_locations::Vector{Tuple{UInt16,UInt16}} = collect(
        (x, y) for x in 1:grid_side_length, y in 1:grid_side_length if
        (x, y) ∉ protected_cell_locations
    )

    cell_lattice = CellLatticeClass.CellLattice(
        cell_lattice_ids,
        quiescent_cell_locations,
        protected_cell_locations,
        smoking_driver_fitness_augmentation,
    )

    cell_compartments_dict::OrderedDict{
        String,
        SpatialCompartmentClass.SpatialCompartment,
    } = OrderedDict(
        "main" => SpatialCompartmentClass.SpatialCompartment(
            randomiser,
            mutation_phylogeny,
            cell_phylogeny,
            CellPhylogenyClass.basal,
            main_compartment_cell_locations,
            cell_lattice,
            1.0,
            use_yearly_counters,
            true,
            true,
            true,
            vary_mutation_rate,
            "main",
        ),
    )

    if quiescent_spacing > 0
        @assert quiescent_protection_coefficient !== nothing
        cell_compartments_dict["quiescent"] =
            SpatialCompartmentClass.SpatialCompartment(
                randomiser,
                mutation_phylogeny,
                cell_phylogeny,
                CellPhylogenyClass.quiescent,
                quiescent_cell_locations,
                cell_lattice,
                quiescent_protection_coefficient,
                use_yearly_counters,
                true,
                false,
                false,
                vary_mutation_rate,
                "quiescent",
                "main",
            )
    end

    if protected_spacing > 0
        cell_compartments_dict["protected"] =
            SpatialCompartmentClass.SpatialCompartment(
                randomiser,
                mutation_phylogeny,
                cell_phylogeny,
                CellPhylogenyClass.basal,
                protected_cell_locations,
                cell_lattice,
                protection_coefficient,
                use_yearly_counters,
                true,
                true,
                true,
                vary_mutation_rate,
                "protected",
            )
    end
    return SpatialCellPopulation(
        randomiser,
        protection_coefficient,
        quiescent_protection_coefficient,
        quiescent_gland_cell_count,
        Ref(0.0),
        Ref(0),
        cell_lattice,
        cell_compartments_dict,
    )
end

function get_quiescent_cell_locations(
    grid_side_length::Int,
    quiescent_spacing::Int,
)::Vector{Tuple{UInt16,UInt16}}
    if quiescent_spacing == 0
        return []
    end
    quiescent_cell_locations::Vector{Tuple{UInt16,UInt16}} = collect(
        (x, y) for x in 1:grid_side_length, y in 1:grid_side_length if
        x % quiescent_spacing == 0 && y % quiescent_spacing == 0
    )
    return quiescent_cell_locations
end

function get_protected_cell_locations(
    grid_side_length::Int,
    protected_spacing::Int,
    protected_region_radius::Int,
)::Vector{Tuple{UInt16,UInt16}}
    if protected_spacing == 0
        return []
    end
    protected_cell_region_centres::Vector{Tuple{UInt16,UInt16}} = collect(
        (x, y) for x in 1:grid_side_length, y in 1:grid_side_length if
        x % protected_spacing == 0 && y % protected_spacing == 0
    )
    protected_cell_locations::Vector{Tuple{UInt16,UInt16}} = []
    for centre in protected_cell_region_centres
        for x in centre[1]-protected_region_radius:centre[1]+protected_region_radius
            for y in centre[2]-protected_region_radius:centre[2]+protected_region_radius
                if x < 1
                    x = grid_side_length + x
                elseif x > grid_side_length
                    x = x - grid_side_length
                end
                if y < 1
                    y = grid_side_length + y
                elseif y > grid_side_length
                    y = y - grid_side_length
                end
                if abs(x - centre[1]) + abs(y - centre[2]) <= protected_region_radius
                    if (x, y) ∉ protected_cell_locations
                        push!(protected_cell_locations, (x, y))
                    end
                end
            end
        end
    end
    return protected_cell_locations
end

function string(spatial_cell_population::SpatialCellPopulation)::String
    compartments_summary::String = join(
        [
            "$compartment_name compartment: " *
            string(
                spatial_cell_population.cell_compartments_dict[compartment_name].initial_cell_number,
            ) *
            " cells, with mean mutation counts: " *
            join(
                [
                    string(
                        mean(
                            CellLatticeClass.get_mutation_counts_list(
                                spatial_cell_population.cell_lattice,
                                compartment.cell_type == CellPhylogenyClass.quiescent,
                                compartment.protection_coefficient < 1.0,
                                driver,
                                smoking_signature,
                            ),
                        ),
                    ) *
                    (smoking_signature ? " smoking" : " non-smoking") *
                    " signature " *
                    (driver ? "drivers" : "passengers") for
                    smoking_signature in [true, false] for driver in [true, false]
                ],
                ", ",
            ) for (compartment_name, compartment) in
            pairs(spatial_cell_population.cell_compartments_dict)
        ],
        "\n\t",
    )
    return "Cell population:\n\t$compartments_summary"
end

function select_next_event(
    spatial_cell_population::SpatialCellPopulation,
    smoking::Bool,
    divisions_per_year::SmokingAffectedParameterClass.SmokingAffectedParameter,
    symmetric_division_prob::Float64,
    immune_death_rate::SmokingAffectedParameterClass.SmokingAffectedParameter,
    ambient_quiescent_division_rate_coeff::SmokingAffectedParameterClass.SmokingAffectedParameter,
)::Tuple{Float64,EventType}
    λ = SmokingAffectedParameterClass.get_value(divisions_per_year, smoking)
    r = symmetric_division_prob
    μ = SmokingAffectedParameterClass.get_value(immune_death_rate, smoking)
    symmetric_differentiation_rate =
        λ * r * spatial_cell_population.cell_lattice.grid_side_length^2
    asymmetric_division_rate =
        λ * (1 - 2 * r) * spatial_cell_population.cell_lattice.grid_side_length^2
    immune_death_rate = μ * spatial_cell_population.total_mutation_count[]
    ambient_quiescent_division_rate = (
        asymmetric_division_rate *
        SmokingAffectedParameterClass.get_protected_value(
            ambient_quiescent_division_rate_coeff,
            smoking,
            spatial_cell_population.quiescent_protection_coefficient,
        ) *
        spatial_cell_population.cell_lattice.true_quiescent_gland_fraction *
        spatial_cell_population.quiescent_gland_cell_count
    )
    # note that this enforces that the total rates are as in the theoretical model

    total_rate =
        symmetric_differentiation_rate +
        asymmetric_division_rate +
        immune_death_rate +
        ambient_quiescent_division_rate
    time_to_next_event = -log(rand(spatial_cell_population.randomiser)) / total_rate
    event_type_rnd = rand(spatial_cell_population.randomiser) * total_rate
    if event_type_rnd < asymmetric_division_rate
        return time_to_next_event, asymmetric_division
    elseif event_type_rnd < asymmetric_division_rate + immune_death_rate
        return time_to_next_event, immune_death
    elseif event_type_rnd <
           asymmetric_division_rate +
           immune_death_rate +
           ambient_quiescent_division_rate
        return time_to_next_event, ambient_quiescent_division
    else
        return time_to_next_event, symmetric_differentiation
    end
end

function select_cell(
    spatial_cell_population::SpatialCellPopulation,
    smoking::Bool,
    event_type::EventType,
    divisions_per_year::SmokingAffectedParameterClass.SmokingAffectedParameter,
    symmetric_division_prob::Float64,
)::Tuple{UInt16,UInt16}
    if event_type == asymmetric_division
        return select_cell_for_asymmetric_division(
            spatial_cell_population,
            smoking,
            divisions_per_year,
        )
        return loc
    elseif event_type == ambient_quiescent_division
        return select_cell_for_ambient_quiescent_division(spatial_cell_population)
    end
    return rejection_sample_cell(
        spatial_cell_population,
        smoking,
        event_type,
        divisions_per_year,
        symmetric_division_prob,
    )
end

function select_cell_for_asymmetric_division(
    spatial_cell_population::SpatialCellPopulation,
    smoking::Bool,
    divisions_per_year::SmokingAffectedParameterClass.SmokingAffectedParameter,
)::Tuple{UInt16,UInt16}
    if !smoking ||
       length(spatial_cell_population.cell_lattice.protected_cell_locations) == 0
        # uniform sample over all cells
        return (
            rand(
                spatial_cell_population.randomiser,
                1:spatial_cell_population.cell_lattice.grid_side_length,
            ),
            rand(
                spatial_cell_population.randomiser,
                1:spatial_cell_population.cell_lattice.grid_side_length,
            ),
        )
    end
    # decide if we're sampling from the protected or main compartment
    protected_cell_count =
        length(spatial_cell_population.cell_lattice.protected_cell_locations)
    protected_division_rate =
        protected_cell_count * (SmokingAffectedParameterClass.get_protected_value(
            divisions_per_year,
            smoking,
            spatial_cell_population.cell_compartments_dict["protected"].protection_coefficient,
        ))
    main_cell_count =
        spatial_cell_population.cell_lattice.grid_side_length^2 - protected_cell_count
    main_division_rate = main_cell_count * divisions_per_year.smoking_value
    if rand(spatial_cell_population.randomiser) <
       protected_division_rate / (protected_division_rate + main_division_rate)
        # sample from protected compartment
        return rand(
            spatial_cell_population.randomiser,
            spatial_cell_population.cell_lattice.protected_cell_locations,
        )
    end
    # sample from main compartment
    return rand(
        spatial_cell_population.randomiser,
        spatial_cell_population.cell_lattice.main_cell_locations,
    )
end

function select_cell_for_ambient_quiescent_division(
    spatial_cell_population::SpatialCellPopulation,
)::Tuple{UInt16,UInt16}
    return sample(
        spatial_cell_population.randomiser,
        spatial_cell_population.cell_lattice.quiescent_cell_locations,
    )
end

function rejection_sample_cell(
    spatial_cell_population::SpatialCellPopulation,
    smoking::Bool,
    event_type::EventType,
    divisions_per_year::SmokingAffectedParameterClass.SmokingAffectedParameter,
    symmetric_division_prob::Float64,
)::Tuple{UInt16,UInt16}
    protection_division_rate_reduction = (
        if smoking &&
           "protected" in keys(spatial_cell_population.cell_compartments_dict)
            SmokingAffectedParameterClass.get_protected_value(
                divisions_per_year,
                smoking,
                spatial_cell_population.cell_compartments_dict["protected"].protection_coefficient,
            ) / divisions_per_year.smoking_value
        else
            1.0
        end
    )
    max_weight = (
        if event_type == symmetric_differentiation
            2 # representing 2r, the maximum possible value of the weight
        else
            spatial_cell_population.cell_lattice.mutation_count_upper_bound[]
        end
    )
    sample_count = 0
    while true
        sample_count += 1
        cell_location_index = rand(
            spatial_cell_population.randomiser,
            1:spatial_cell_population.cell_lattice.grid_side_length^2,
        )
        cell_location = CellLatticeClass.get_cell_location(
            spatial_cell_population.cell_lattice,
            cell_location_index,
        )
        relative_weight =
            get_cell_weight(
                spatial_cell_population,
                cell_location,
                smoking,
                event_type,
                symmetric_division_prob,
                protection_division_rate_reduction,
            ) / max_weight
        if rand(spatial_cell_population.randomiser) < relative_weight
            return cell_location
        end
        if sample_count > 100000
            normalised_fitnesses =
                (
                    if (
                        smoking &&
                        spatial_cell_population.cell_lattice.smoking_fitnesses !==
                        nothing
                    )
                        spatial_cell_population.cell_lattice.smoking_fitnesses
                    else
                        spatial_cell_population.cell_lattice.fitnesses
                    end
                ) .- spatial_cell_population.current_normalisation_constant[]
            error(
                "RejectionSamplingError in $event_type; distribution of fitnesses " *
                "(after normalisation by $(spatial_cell_population.current_normalisation_constant[])," *
                "mean $(mean(normalised_fitnesses))): " *
                string(
                    quantile(
                        normalised_fitnesses,
                        [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                    ),
                ),
            )
        end
    end
end

function get_cell_weight(
    spatial_cell_population::SpatialCellPopulation,
    cell_location::Tuple{UInt16,UInt16},
    smoking::Bool,
    event_type::EventType,
    symmetric_division_prob::Float64,
    protected_division_rate_reduction::Float64,
)::Float64
    if event_type == symmetric_differentiation
        cell_weight = (
            if smoking &&
               spatial_cell_population.cell_lattice.smoking_fitnesses !== nothing
                (
                    1 - CurrentFitnessClass.project_fitness_value(
                        spatial_cell_population.cell_lattice.smoking_fitnesses[cell_location...] -
                        spatial_cell_population.current_normalisation_constant[],
                        1.0,
                    )
                )
            else
                (
                    1 - CurrentFitnessClass.project_fitness_value(
                        spatial_cell_population.cell_lattice.fitnesses[cell_location...] -
                        spatial_cell_population.current_normalisation_constant[],
                        1.0,
                    )
                )
            end
        )
        if spatial_cell_population.cell_lattice.protected_mask[cell_location...]
            return cell_weight * protected_division_rate_reduction
        end
        return cell_weight
    elseif event_type == immune_death
        return spatial_cell_population.cell_lattice.mutation_counts[cell_location...]
    end
    throw(ArgumentError("event_type must be symmetric_differentiation or immune_death"))
end

function replace_cell!(
    spatial_cell_population::SpatialCellPopulation,
    removed_cell_location::Tuple{UInt16,UInt16},
    smoking::Bool,
    divisions_per_year::SmokingAffectedParameterClass.SmokingAffectedParameter,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    symmetric_division_prob::Float64,
    fitness_change_scale::Float64,
    expected_mutations_per_division::Dict{Bool,Dict{Bool,Float64}},
    quiescent_division_rate_coeff::SmokingAffectedParameterClass.SmokingAffectedParameter,
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    record_number::UInt16,
    event_type::EventType,
)::Nothing
    cell_id = spatial_cell_population.cell_lattice[removed_cell_location...]
    removed_cell_compartment_name = (
        if spatial_cell_population.cell_lattice.protected_mask[removed_cell_location...]
            "protected"
        else
            "main"
        end
    )
    SpatialCompartmentClass.remove_cell!(
        spatial_cell_population.cell_compartments_dict[removed_cell_compartment_name],
        cell_id,
        record_number,
    )
    if spatial_cell_population.cell_compartments_dict[removed_cell_compartment_name].yearly_counters !==
       nothing
        if event_type == symmetric_differentiation
            spatial_cell_population.cell_compartments_dict[removed_cell_compartment_name].yearly_counters.this_year_differentiate_count +=
                1
        elseif event_type == immune_death
            spatial_cell_population.cell_compartments_dict[removed_cell_compartment_name].yearly_counters.this_year_immune_death_count +=
                1
        end
    end

    dividing_compartment, dividing_cell_id, dividing_cell_location =
        sample_replacing_cell(
            spatial_cell_population,
            removed_cell_location,
            smoking,
            divisions_per_year,
            quiescent_division_rate_coeff,
            symmetric_division_prob,
        )

    randomise_mutation!(
        spatial_cell_population,
        dividing_compartment,
        dividing_cell_id,
        dividing_cell_location,
        smoking,
        mutation_driver_probability,
        fitness_change_probability,
        fitness_change_scale,
        expected_mutations_per_division,
        smoking_driver_fitness_augmentation,
        mutation_rate_multiplier_shape,
        record_number,
    )

    new_cell_id = SpatialCompartmentClass.symmetric_divide_cell!(
        dividing_compartment,
        dividing_cell_id,
        record_number,
        removed_cell_location,
    )
    if dividing_compartment.name == "quiescent"
        quiescent_spacing =
            spatial_cell_population.cell_lattice.quiescent_cell_locations[1][1]
        spatial_cell_population.cell_lattice.quiescent_division_counts[(
            dividing_cell_location .÷ quiescent_spacing
        )...] += 1
        spatial_cell_population.total_mutation_count[] +=
            spatial_cell_population.cell_lattice.quiescent_mutation_counts[(
                dividing_cell_location .÷ quiescent_spacing
            )...]
    else
        spatial_cell_population.cell_lattice.division_counts[removed_cell_location...] +=
            1
        spatial_cell_population.total_mutation_count[] +=
            spatial_cell_population.cell_lattice.mutation_counts[dividing_cell_location...]
    end
    spatial_cell_population.total_mutation_count[] -=
        spatial_cell_population.cell_lattice.mutation_counts[removed_cell_location...]

    spatial_cell_population.current_normalisation_constant[] +=
        CellLatticeClass.calculate_fitness_difference(
            spatial_cell_population.cell_lattice,
            dividing_cell_location,
            dividing_compartment.name == "quiescent",
            removed_cell_location,
            smoking,
        )
    CellLatticeClass.replace_cell!(
        spatial_cell_population.cell_lattice,
        removed_cell_location,
        dividing_cell_location,
        dividing_compartment.name == "quiescent",
        new_cell_id,
    )
    return nothing
end

function sample_replacing_cell(
    spatial_cell_population::SpatialCellPopulation,
    removed_cell_location::Tuple{UInt16,UInt16},
    smoking::Bool,
    divisions_per_year::SmokingAffectedParameterClass.SmokingAffectedParameter,
    quiescent_division_rate_coeff::SmokingAffectedParameterClass.SmokingAffectedParameter,
    symmetric_division_prob::Float64,
)::Tuple{SpatialCompartmentClass.SpatialCompartment,UInt64,Tuple{UInt16,UInt16}}
    neighbour_cell_ids, neighbour_cell_locations, neighbour_cell_compartment_names =
        CellLatticeClass.get_neighbour_cells(
            spatial_cell_population.cell_lattice,
            removed_cell_location,
        )

    neighbour_cell_relative_symmetric_division_rates = [
        (
            if neighbour_cell_compartment_name == "quiescent"
                SmokingAffectedParameterClass.get_protected_value(
                    quiescent_division_rate_coeff,
                    smoking,
                    spatial_cell_population.quiescent_protection_coefficient,
                ) * (
                    1 + CurrentFitnessClass.project_fitness_value(
                        spatial_cell_population.cell_lattice.quiescent_fitnesses[(
                            neighbour_cell_location .÷
                            spatial_cell_population.cell_lattice.quiescent_cell_locations[1][1]
                        )...] -
                        spatial_cell_population.current_normalisation_constant[],
                        1.0,
                    )
                )
            else
                1 + CurrentFitnessClass.project_fitness_value(
                    (
                        if smoking &&
                           spatial_cell_population.cell_lattice.smoking_fitnesses !==
                           nothing
                            spatial_cell_population.cell_lattice.smoking_fitnesses[neighbour_cell_location...]
                        else
                            spatial_cell_population.cell_lattice.fitnesses[neighbour_cell_location...]
                        end
                    ) - spatial_cell_population.current_normalisation_constant[],
                    1.0,
                )
            end
        ) for (neighbour_cell_location, neighbour_cell_compartment_name) in
        zip(neighbour_cell_locations, neighbour_cell_compartment_names)
    ]
    if smoking
        # then the different protection coefficients of the compartments are
        # relevant, so we need to multiply
        for (i, compartment_name) in enumerate(neighbour_cell_compartment_names)
            if (
                spatial_cell_population.cell_compartments_dict[compartment_name].protection_coefficient !=
                1.0
            )
                neighbour_cell_relative_symmetric_division_rates[i] *=
                    SmokingAffectedParameterClass.get_protected_value(
                        divisions_per_year,
                        true,
                        spatial_cell_population.cell_compartments_dict[compartment_name].protection_coefficient,
                    ) / divisions_per_year.smoking_value
            end
        end
    end
    dividing_neighbour_index = sample(
        spatial_cell_population.randomiser,
        1:length(neighbour_cell_ids),
        Weights(neighbour_cell_relative_symmetric_division_rates),
    )
    dividing_compartment =
        spatial_cell_population.cell_compartments_dict[neighbour_cell_compartment_names[dividing_neighbour_index]]
    dividing_cell_id = neighbour_cell_ids[dividing_neighbour_index]
    dividing_cell_location = neighbour_cell_locations[dividing_neighbour_index]
    return dividing_compartment, dividing_cell_id, dividing_cell_location
end

function get_direction(
    dividing_cell_location::Tuple{UInt16,UInt16},
    removed_cell_location::Tuple{UInt16,UInt16},
)::String
    if dividing_cell_location[1] == removed_cell_location[1]
        if dividing_cell_location[2] == removed_cell_location[2] + 1
            return "up"
        elseif dividing_cell_location[2] == removed_cell_location[2] - 1
            return "down"
        else
            return "v_edge"
        end
    elseif dividing_cell_location[2] == removed_cell_location[2]
        if dividing_cell_location[1] == removed_cell_location[1] + 1
            return "right"
        elseif dividing_cell_location[1] == removed_cell_location[1] - 1
            return "left"
        else
            return "h_edge"
        end
    end
    return "error"
end

function asymmetric_divide_cell!(
    spatial_cell_population::SpatialCellPopulation,
    dividing_cell_location::Tuple{UInt16,UInt16},
    smoking::Bool,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    fitness_change_scale::Float64,
    expected_mutations_per_division::Dict{Bool,Dict{Bool,Float64}},
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    record_number::UInt16,
)::Nothing
    spatial_cell_population.cell_lattice.division_counts[dividing_cell_location...] += 1
    dividing_cell_id = spatial_cell_population.cell_lattice[dividing_cell_location...]
    dividing_compartment_name = (
        if spatial_cell_population.cell_lattice.protected_mask[dividing_cell_location...]
            "protected"
        else
            "main"
        end
    )
    randomise_mutation!(
        spatial_cell_population,
        spatial_cell_population.cell_compartments_dict[dividing_compartment_name],
        dividing_cell_id,
        dividing_cell_location,
        smoking,
        mutation_driver_probability,
        fitness_change_probability,
        fitness_change_scale,
        expected_mutations_per_division,
        smoking_driver_fitness_augmentation,
        mutation_rate_multiplier_shape,
        record_number,
    )
    return nothing
end

function ambient_quiescent_divide_cell!(
    spatial_cell_population::SpatialCellPopulation,
    event_cell_location::Tuple{UInt16,UInt16},
    smoking::Bool,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    fitness_change_scale::Float64,
    expected_mutations_per_division::Dict{Bool,Dict{Bool,Float64}},
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    current_record_number::UInt16,
)::Nothing
    quiescent_spacing =
        spatial_cell_population.cell_lattice.quiescent_cell_locations[1][1]
    mutating_cell_id = spatial_cell_population.cell_lattice.quiescent_cell_ids[(
        event_cell_location .÷ quiescent_spacing
    )...]
    spatial_cell_population.cell_lattice.quiescent_division_counts[(
        event_cell_location .÷ quiescent_spacing
    )...] += 1
    return randomise_mutation!(
        spatial_cell_population,
        spatial_cell_population.cell_compartments_dict["quiescent"],
        mutating_cell_id,
        event_cell_location,
        smoking,
        mutation_driver_probability,
        fitness_change_probability,
        fitness_change_scale,
        expected_mutations_per_division,
        smoking_driver_fitness_augmentation,
        mutation_rate_multiplier_shape,
        current_record_number,
        spatial_cell_population.quiescent_gland_cell_count,
    )
end

function randomise_mutation!(
    spatial_cell_population::SpatialCellPopulation,
    mutating_compartment::SpatialCompartmentClass.SpatialCompartment,
    mutating_cell_id::UInt64,
    mutating_cell_location::Tuple{UInt16,UInt16},
    smoking::Bool,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    fitness_change_scale::Float64,
    expected_mutations_per_division::Dict{Bool,Dict{Bool,Float64}},
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    record_number::UInt16,
    fixation_population_size::Union{Int,Nothing} = nothing,
)::Nothing
    (
        driver_smoking_signature_mutation_count,
        passenger_smoking_signature_mutation_count,
        driver_non_smoking_signature_mutation_count,
        passenger_non_smoking_signature_mutation_count,
        smoking_dependent_fitness_change,
        smoking_smoking_dependent_fitness_change,
        smoking_independent_fitness_change,
        smoking_smoking_independent_fitness_change,
    ) = CompartmentClass.randomise_mutation!(
        mutating_compartment,
        mutating_cell_id,
        smoking,
        mutation_driver_probability,
        fitness_change_probability,
        fitness_change_scale,
        expected_mutations_per_division[mutating_compartment.protection_coefficient!=1],
        smoking_driver_fitness_augmentation,
        mutation_rate_multiplier_shape,
        record_number,
        spatial_cell_population.cell_lattice.smoking_fitnesses !== nothing,
        fixation_population_size,
    )
    total_fitness_change =
        smoking_dependent_fitness_change + smoking_independent_fitness_change
    total_smoking_fitness_change = (
        smoking_smoking_dependent_fitness_change +
        smoking_smoking_independent_fitness_change
    )
    mutation_count = (
        driver_smoking_signature_mutation_count +
        passenger_smoking_signature_mutation_count +
        driver_non_smoking_signature_mutation_count +
        passenger_non_smoking_signature_mutation_count
    )
    # quiescent cells don't contribute to the normalisation constant, as they don't
    # differentiate
    if mutating_compartment.name != "quiescent"
        spatial_cell_population.current_normalisation_constant[] += (
            if smoking &&
               spatial_cell_population.cell_lattice.smoking_fitnesses !== nothing
                total_smoking_fitness_change /
                spatial_cell_population.cell_lattice.grid_side_length^2
            else
                total_fitness_change /
                spatial_cell_population.cell_lattice.grid_side_length^2
            end
        )
    end
    CellLatticeClass.increment_mutation_count(
        spatial_cell_population.cell_lattice,
        mutating_cell_location,
        mutation_count,
        driver_smoking_signature_mutation_count,
        passenger_smoking_signature_mutation_count,
        driver_non_smoking_signature_mutation_count,
        passenger_non_smoking_signature_mutation_count,
        mutating_compartment.name == "quiescent",
    )
    if mutating_compartment.name == "quiescent"
        quiescent_spacing =
            spatial_cell_population.cell_lattice.quiescent_cell_locations[1][1]
        location_on_quiescent_grid = mutating_cell_location .÷ quiescent_spacing
        spatial_cell_population.cell_lattice.quiescent_fitnesses[(
            location_on_quiescent_grid
        )...] += total_fitness_change
        if spatial_cell_population.cell_lattice.quiescent_smoking_fitnesses !== nothing
            spatial_cell_population.cell_lattice.quiescent_smoking_fitnesses[(
                location_on_quiescent_grid
            )...] += total_smoking_fitness_change
        end
    else
        spatial_cell_population.total_mutation_count[] += mutation_count
        spatial_cell_population.cell_lattice.fitnesses[mutating_cell_location...] +=
            total_fitness_change
        if spatial_cell_population.cell_lattice.smoking_fitnesses !== nothing
            spatial_cell_population.cell_lattice.smoking_fitnesses[mutating_cell_location...] +=
                total_smoking_fitness_change
        end
    end
    return nothing
end

function recalibrate_current_normalisation_constant!(
    spatial_cell_population::SpatialCellPopulation,
    smoking::Bool,
)::Float64
    previous_normalisation_constant =
        spatial_cell_population.current_normalisation_constant[]
    total_fitness = (
        if smoking &&
           spatial_cell_population.cell_lattice.smoking_fitnesses !== nothing
            sum(spatial_cell_population.cell_lattice.smoking_fitnesses)
        else
            sum(spatial_cell_population.cell_lattice.fitnesses)
        end
    )
    spatial_cell_population.current_normalisation_constant[] =
        total_fitness / spatial_cell_population.cell_lattice.grid_side_length^2
    return previous_normalisation_constant -
           spatial_cell_population.current_normalisation_constant[]
end

function record_fitness_summary(
    spatial_cell_population::SpatialCellPopulation,
    fitness_summaries::Records.Record,
    current_record_number::UInt16,
)::Nothing
    normalised_fitnesses =
        spatial_cell_population.cell_lattice.fitnesses .-
        spatial_cell_population.current_normalisation_constant[]
    normalised_smoking_fitnesses =
        (
            if spatial_cell_population.cell_lattice.smoking_fitnesses !== nothing
                spatial_cell_population.cell_lattice.smoking_fitnesses
            else
                spatial_cell_population.cell_lattice.fitnesses
            end
        ) .- spatial_cell_population.current_normalisation_constant[]
    if spatial_cell_population.cell_lattice.quiescent_fitnesses !== nothing
        normalised_quiescent_fitnesses =
            spatial_cell_population.cell_lattice.quiescent_fitnesses .-
            spatial_cell_population.current_normalisation_constant[]
        normalised_quiescent_smoking_fitnesses =
            (
                if spatial_cell_population.cell_lattice.quiescent_smoking_fitnesses !==
                   nothing
                    spatial_cell_population.cell_lattice.quiescent_smoking_fitnesses
                else
                    spatial_cell_population.cell_lattice.quiescent_fitnesses
                end
            ) .- spatial_cell_population.current_normalisation_constant[]
    end
    Records.record_fitness_summary(
        fitness_summaries,
        current_record_number,
        "total",
        mean(normalised_smoking_fitnesses),
        mean(normalised_fitnesses),
        std(normalised_smoking_fitnesses),
        std(normalised_fitnesses),
        spatial_cell_population.current_normalisation_constant[],
    )
    Records.record_fitness_summary(
        fitness_summaries,
        current_record_number,
        "main",
        mean(
            normalised_smoking_fitnesses[.!spatial_cell_population.cell_lattice.protected_mask],
        ),
        mean(
            normalised_fitnesses[.!spatial_cell_population.cell_lattice.protected_mask],
        ),
        std(
            normalised_smoking_fitnesses[.!spatial_cell_population.cell_lattice.protected_mask],
        ),
        std(
            normalised_fitnesses[.!spatial_cell_population.cell_lattice.protected_mask],
        ),
        spatial_cell_population.current_normalisation_constant[],
    )
    if any(spatial_cell_population.cell_lattice.protected_mask)
        Records.record_fitness_summary(
            fitness_summaries,
            current_record_number,
            "protected",
            mean(
                normalised_smoking_fitnesses[spatial_cell_population.cell_lattice.protected_mask],
            ),
            mean(
                normalised_fitnesses[spatial_cell_population.cell_lattice.protected_mask],
            ),
            std(
                normalised_smoking_fitnesses[spatial_cell_population.cell_lattice.protected_mask],
            ),
            std(
                normalised_fitnesses[spatial_cell_population.cell_lattice.protected_mask],
            ),
            spatial_cell_population.current_normalisation_constant[],
        )
    end
    if spatial_cell_population.cell_lattice.quiescent_fitnesses !== nothing
        Records.record_fitness_summary(
            fitness_summaries,
            current_record_number,
            "quiescent",
            mean(normalised_quiescent_smoking_fitnesses),
            mean(normalised_quiescent_fitnesses),
            std(normalised_quiescent_smoking_fitnesses),
            std(normalised_quiescent_fitnesses),
            spatial_cell_population.current_normalisation_constant[],
        )
    end
    return nothing
end

function record_mutational_burden(
    spatial_cell_population::SpatialCellPopulation,
    mutational_burden_record::Records.Record,
    current_record_number::UInt16,
)::Nothing
    CellLatticeClass.record_mutational_burden(
        spatial_cell_population.cell_lattice,
        mutational_burden_record,
        current_record_number,
    )
    return nothing
end

function get_final_mutational_burden(
    spatial_cell_population::SpatialCellPopulation,
    smoking_signature::Bool,
)::Vector{UInt32}
    mutations_array::Matrix{UInt32} = (
        if smoking_signature
            spatial_cell_population.cell_lattice.mutation_counts.driver_smoking_signature_mutations +
            spatial_cell_population.cell_lattice.mutation_counts.passenger_smoking_signature_mutations
        else
            spatial_cell_population.cell_lattice.mutation_counts.total_mutations
        end
    )
    mutation_counts = reshape(mutations_array, :)
    if "quiescent" in spatial_cell_population.cell_compartments_dict.keys &&
       spatial_cell_population.cell_compartments_dict["quiescent"].to_be_counted
        quiescent_mutations_array = (
            if smoking_signature
                spatial_cell_population.cell_lattice.quiescent_mutation_counts.driver_smoking_signature_mutations +
                spatial_cell_population.cell_lattice.quiescent_mutation_counts.passenger_smoking_signature_mutations
            else
                spatial_cell_population.cell_lattice.quiescent_mutation_counts.total_mutations
            end
        )
        mutation_counts = vcat(mutation_counts, reshape(quiescent_mutations_array, :))
    end
    return mutation_counts
end

end
