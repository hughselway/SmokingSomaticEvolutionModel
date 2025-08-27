module CompartmentClass

using Random
using PoissonRandom
import Base: getindex, string

using ..CellPhylogenyClass
using ..MutationPhylogenyClass
using ..SmokingAffectedParameterClass
using ..CurrentFitnessClass
using ..Records

mutable struct YearlyCounters
    this_year_new_cell_count::Int
    this_year_differentiate_count::Int
    this_year_immune_death_count::Int
end

YearlyCounters() = YearlyCounters(0, 0, 0)

function reset!(yc::YearlyCounters)::Nothing
    yc.this_year_new_cell_count = 0
    yc.this_year_differentiate_count = 0
    yc.this_year_immune_death_count = 0
    return nothing
end

abstract type AbstractCompartment end

struct Compartment <: AbstractCompartment
    alive_cell_ids::Vector{UInt64}
    randomiser::Random.Xoshiro
    mutation_phylogeny::Union{MutationPhylogenyClass.MutationPhylogeny,Nothing}
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        CellPhylogenyClass.CellPhylogenyClass.EmptyCellPhylogeny,
    }
    cell_type::CellPhylogenyClass.CellType
    initial_cell_number::Int
    protection_coefficient::Float64
    to_be_counted::Bool
    affected_by_immune_system::Bool
    included_in_competition::Bool
    destination_compartment_name::Union{String,Nothing}
    yearly_counters::Union{YearlyCounters,Nothing}
end

function Compartment(
    randomiser::Random.Xoshiro,
    mutation_phylogeny::Union{MutationPhylogenyClass.MutationPhylogeny,Nothing},
    cell_phylogeny::Union{
        CellPhylogenyClass.CellPhylogeny,
        CellPhylogenyClass.CellPhylogenyClass.EmptyCellPhylogeny,
    },
    cell_type::CellPhylogenyClass.CellType,
    initial_cell_number::Int,
    protection_coefficient::Float64,
    use_yearly_counters::Bool,
    to_be_counted::Bool,
    affected_by_immune_system::Bool,
    included_in_competition::Bool,
    vary_mutation_rate::Bool,
    destination_compartment_name::Union{String,Nothing} = nothing,
)::Compartment
    alive_cell_ids = CellPhylogenyClass.initialise_cells!(
        cell_phylogeny,
        initial_cell_number,
        vary_mutation_rate,
    )
    yearly_counters = use_yearly_counters ? YearlyCounters() : nothing
    return Compartment(
        alive_cell_ids,
        randomiser,
        mutation_phylogeny,
        cell_phylogeny,
        cell_type,
        initial_cell_number,
        protection_coefficient,
        to_be_counted,
        affected_by_immune_system,
        included_in_competition,
        destination_compartment_name,
        yearly_counters,
    )
end

function string(compartment::Compartment)::String
    return (
        "$(compartment.cell_type) cell compartment with " *
        "$(cell_count(compartment)) cells, " *
        "protection coefficient $(compartment.protection_coefficient), " *
        (compartment.to_be_counted ? "" : "not ") *
        "to be counted, " *
        (compartment.affected_by_immune_system ? "" : "not ") *
        "affected by immune system, " *
        (compartment.included_in_competition ? "" : "not ") *
        "included in competition" *
        (
            if compartment.destination_compartment_name === nothing
                ""
            else
                ", destination compartment $(compartment.destination_compartment_name)"
            end
        )
    )
end

function cell_list(
    compartment::AbstractCompartment,
)::Vector{CellPhylogenyClass.AbstractCell}
    return [
        compartment.cell_phylogeny.cell_list[cell_id] for
        cell_id in compartment.alive_cell_ids
    ]
end

function getindex(
    compartment::AbstractCompartment,
    cell_id::UInt64,
)::CellPhylogenyClass.AbstractCell
    @assert cell_id in compartment.alive_cell_ids
    return compartment.cell_phylogeny.cell_list[cell_id]
end

function yearly_stats_string(compartment::Compartment)::String
    return (
        "cell count: $(cell_count(compartment)), " *
        yearly_counters_string(compartment)
    )
end

function yearly_counters_string(compartment::AbstractCompartment)::String
    @assert compartment.yearly_counters !== nothing
    return (
        "new cells: $(compartment.yearly_counters.this_year_new_cell_count), " *
        (
            if compartment.cell_type == CellPhylogenyClass.basal
                "differentiated cells: " *
                "$(compartment.yearly_counters.this_year_differentiate_count), "
            else
                ""
            end
        ) *
        "immune deaths: $(compartment.yearly_counters.this_year_immune_death_count)"
    )
end

function cell_count(compartment::Compartment)::Int
    return length(compartment.alive_cell_ids)
end

function cell_count(compartment::AbstractCompartment)::Int
    return compartment.initial_cell_number
end

function remove_cells!(
    compartment::Compartment,
    cell_ids::Union{Vector{UInt64}},
    step::UInt16,
)::Nothing
    @assert all([cell_id in compartment.alive_cell_ids for cell_id in cell_ids])
    for cell_id in cell_ids
        CellPhylogenyClass.set_step_of_death_or_removal!(
            compartment.cell_phylogeny,
            cell_id,
            step,
        )
    end
    deleteat!(
        compartment.alive_cell_ids,
        findall(x -> x in cell_ids, compartment.alive_cell_ids),
    )
    return nothing
end

function apply_immune_system!(
    compartment::Compartment,
    immune_death_prob_per_mutation::Float64,
    step_number::UInt16,
)::Nothing
    if !compartment.affected_by_immune_system
        return
    end
    removing_cell_ids = Vector{UInt64}(undef, cell_count(compartment))
    removing_cell_count::Int = 0
    for cell_id in compartment.alive_cell_ids
        # TODO: switch from linear to sigmoid (as an option)
        if rand(compartment.randomiser) < (
            immune_death_prob_per_mutation *
            MutationPhylogenyClass.get_total_mutation_count(
                compartment.mutation_phylogeny,
                # compartment.cell_phylogeny[cell_id].mutational_profile_id[],
                compartment.cell_phylogeny.mutational_profile_ids[cell_id],
            )
        )
            removing_cell_count += 1
            removing_cell_ids[removing_cell_count] = cell_id
        end
    end
    if removing_cell_count > 0
        remove_cells!(
            compartment,
            removing_cell_ids[1:removing_cell_count],
            step_number,
        )
        if compartment.yearly_counters !== nothing
            compartment.yearly_counters.this_year_immune_death_count +=
                removing_cell_count
        end
    end
    return nothing
end

function record_mutational_burden(
    compartment::AbstractCompartment,
    mutational_burden_record::Records.Record,
    step_number::UInt16,
    compartment_name::String,
)::Nothing
    for cell_id in compartment.alive_cell_ids
        mutational_profile_id =
            compartment.cell_phylogeny.mutational_profile_ids[cell_id]
        Records.record_mutational_burden!(
            mutational_burden_record,
            step_number,
            cell_id,
            MutationPhylogenyClass.mutation_count(
                compartment.mutation_phylogeny,
                mutational_profile_id,
                true,
                false,
            ),
            MutationPhylogenyClass.mutation_count(
                compartment.mutation_phylogeny,
                mutational_profile_id,
                true,
                true,
            ),
            MutationPhylogenyClass.mutation_count(
                compartment.mutation_phylogeny,
                mutational_profile_id,
                false,
                false,
            ),
            MutationPhylogenyClass.mutation_count(
                compartment.mutation_phylogeny,
                mutational_profile_id,
                false,
                true,
            ),
            # TODO: record divisions rather than recalculating at each step
            CellPhylogenyClass.count_divisions(compartment.cell_phylogeny, cell_id),
            compartment_name,
        )
    end
    return nothing
end

function get_mutation_counts_list(
    compartment::AbstractCompartment,
    driver::Union{Bool,Nothing},
    smoking_signature::Union{Bool,Nothing},
)::Vector{Int}
    return Vector{Int}([
        sum([
            MutationPhylogenyClass.mutation_count(
                compartment.mutation_phylogeny,
                compartment.cell_phylogeny.mutational_profile_ids[cell_id],
                driver,
                smoking_signature,
            ) for driver in (driver === nothing ? [true, false] : [driver]) for
            smoking_signature in
            (smoking_signature === nothing ? [true, false] : [smoking_signature])
        ]) for cell_id in compartment.alive_cell_ids
    ])
end

function get_total_fitness(compartment::Compartment, smoking::Bool)
    return sum(
        cell_id -> CurrentFitnessClass.get_value(
            compartment[cell_id].current_fitness,
            smoking,
        ),
        compartment.alive_cell_ids,
    )
end

function divide_cells!(
    compartment::Compartment,
    smoking::Bool,
    expected_divisions_each_step::SmokingAffectedParameterClass.SmokingAffectedParameter,
    expected_mutations_per_division::SmokingAffectedParameterClass.SmokingAffectedParameter,
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    symmetric_division_prob::Float64,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    fitness_change_scale::Float64,
    step_number::UInt16,
)::Vector{UInt64}
    new_cell_ids = UInt64[]
    differentiating_cell_ids = UInt64[]

    expected_divisions_this_step = SmokingAffectedParameterClass.get_protected_value(
        expected_divisions_each_step,
        smoking,
        compartment.protection_coefficient,
    )
    expected_mutations_per_division_this_step = Dict{Bool,Float64}(
        true => SmokingAffectedParameterClass.get_protected_value(
            expected_mutations_per_division,
            true,
            compartment.protection_coefficient,
        ),
        false => SmokingAffectedParameterClass.get_protected_value(
            expected_mutations_per_division,
            false,
            compartment.protection_coefficient,
        ),
    )

    dividing_cell_ids = [
        cell_id for cell_id in compartment.alive_cell_ids for _ in
        1:PoissonRandom.pois_rand(compartment.randomiser, expected_divisions_this_step)
    ]

    for dividing_cell_id in dividing_cell_ids
        if dividing_cell_id in differentiating_cell_ids
            continue
        end
        randomise_mutation!(
            compartment,
            dividing_cell_id,
            smoking,
            mutation_driver_probability,
            fitness_change_probability,
            fitness_change_scale,
            expected_mutations_per_division_this_step,
            smoking_driver_fitness_augmentation,
            mutation_rate_multiplier_shape,
            step_number,
        )
        if compartment.cell_type == CellPhylogenyClass.quiescent
            division_type = CellPhylogenyClass.symmetric_division
        else
            division_type_randomiser = rand(compartment.randomiser)
            if (
                2 * symmetric_division_prob <
                division_type_randomiser <
                1 - (2 * symmetric_division_prob)
            )
                division_type = CellPhylogenyClass.asymmetric_division
            else
                division_type = randomise_division_type(
                    compartment,
                    dividing_cell_id,
                    symmetric_division_prob,
                    smoking,
                    division_type_randomiser,
                )
            end
        end
        if division_type == CellPhylogenyClass.symmetric_division
            new_cell_id = CellPhylogenyClass.symmetric_divide!(
                compartment.cell_phylogeny,
                dividing_cell_id,
                step_number,
            )
            if compartment.yearly_counters !== nothing
                compartment.yearly_counters.this_year_new_cell_count += 1
            end
            if compartment.destination_compartment_name === nothing
                push!(compartment.alive_cell_ids, new_cell_id)
            else
                push!(new_cell_ids, new_cell_id)
            end
        elseif division_type == CellPhylogenyClass.symmetric_differentiation
            push!(differentiating_cell_ids, dividing_cell_id)
            if compartment.yearly_counters !== nothing
                compartment.yearly_counters.this_year_differentiate_count += 1
            end
        elseif division_type != CellPhylogenyClass.asymmetric_division
            throw(ArgumentError("division type $division_type is not valid"))
        end
    end
    remove_cells!(compartment, differentiating_cell_ids, step_number)
    return new_cell_ids
end

function randomise_mutation!(
    compartment::AbstractCompartment,
    cell_id::UInt64,
    smoking::Bool,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    fitness_change_scale::Float64,
    expected_mutations_per_division::Dict{Bool,Float64},
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    step::UInt16,
    calculate_smoking_fitness_change::Bool = false,
    fixation_population_size::Union{Int,Nothing} = nothing,
)::Tuple{Int,Int,Int,Int,Float64,Float64,Float64,Float64}
    smoking_independent_mutation_count = PoissonRandom.pois_rand(
        compartment.randomiser,
        expected_mutations_per_division[false] * (
            if mutation_rate_multiplier_shape === nothing
                1.0
            else
                compartment[cell_id].mutation_rate_multiplier[]
            end
        ),
    )
    smoking_dependent_mutation_count = (
        if smoking
            PoissonRandom.pois_rand(
                compartment.randomiser,
                (
                    expected_mutations_per_division[true] -
                    expected_mutations_per_division[false]
                ) * (
                    if mutation_rate_multiplier_shape === nothing
                        1.0
                    else
                        compartment[cell_id].mutation_rate_multiplier[]
                    end
                ),
            )
        else
            0
        end
    )
    driver_non_smoking_signature_mutation_count::Int = 0
    passenger_non_smoking_signature_mutation_count::Int = 0
    smoking_independent_fitness_change::Float64 = 0.0
    smoking_smoking_independent_fitness_change::Float64 = 0.0
    # dodgy nomenclature: this is the change to the fitness of cells during smoking
    # (which can be different in the smoking_driver module) that is brought about by
    # non-smoking-signature mutations
    for _ in 1:smoking_independent_mutation_count
        fitness_change = mutate!(
            compartment,
            cell_id,
            mutation_driver_probability,
            fitness_change_probability,
            step,
            fitness_change_scale,
            smoking_driver_fitness_augmentation,
            mutation_rate_multiplier_shape,
            false,
            fixation_population_size,
        )
        if fitness_change === nothing
            @assert fixation_population_size !== nothing
            smoking_independent_mutation_count -= 1
            continue
        end
        if fitness_change > 0
            driver_non_smoking_signature_mutation_count += 1
        else
            passenger_non_smoking_signature_mutation_count += 1
        end
        smoking_independent_fitness_change += fitness_change
        if calculate_smoking_fitness_change
            smoking_smoking_independent_fitness_change +=
                CurrentFitnessClass.get_smoking_fitness_change(
                    fitness_change,
                    smoking_driver_fitness_augmentation,
                    compartment.protection_coefficient,
                )
        end
    end
    driver_smoking_signature_mutation_count = 0
    passenger_smoking_signature_mutation_count = 0
    smoking_dependent_fitness_change = 0.0
    smoking_smoking_dependent_fitness_change = 0.0
    if smoking
        for _ in 1:smoking_dependent_mutation_count
            fitness_change = mutate!(
                compartment,
                cell_id,
                mutation_driver_probability,
                fitness_change_probability,
                step,
                fitness_change_scale,
                smoking_driver_fitness_augmentation,
                mutation_rate_multiplier_shape,
                true,
                fixation_population_size,
            )
            if fitness_change === nothing
                @assert fixation_population_size !== nothing
                smoking_dependent_mutation_count -= 1
                continue
            end
            if fitness_change > 0
                driver_smoking_signature_mutation_count += 1
            else
                passenger_smoking_signature_mutation_count += 1
            end
            smoking_dependent_fitness_change += fitness_change
            if calculate_smoking_fitness_change
                smoking_smoking_dependent_fitness_change +=
                    CurrentFitnessClass.get_smoking_fitness_change(
                        fitness_change,
                        smoking_driver_fitness_augmentation,
                        compartment.protection_coefficient,
                    )
            end
        end
    end
    return (
        driver_smoking_signature_mutation_count,
        passenger_smoking_signature_mutation_count,
        driver_non_smoking_signature_mutation_count,
        passenger_non_smoking_signature_mutation_count,
        smoking_dependent_fitness_change,
        smoking_smoking_dependent_fitness_change,
        smoking_independent_fitness_change,
        smoking_smoking_independent_fitness_change,
    )
end

function mutate!(
    compartment::AbstractCompartment,
    cell_id::UInt64,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    step::UInt16,
    fitness_change_scale::Float64,
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    smoking_signature::Bool,
    fixation_population_size::Union{Int,Nothing},
)::Union{Float64,Nothing}
    fitness_change = CurrentFitnessClass.get_fitness_change(
        compartment.randomiser,
        fitness_change_scale,
        mutation_driver_probability,
        fitness_change_probability,
    )
    if fixation_population_size !== nothing &&
       !mutation_fixates(compartment, fitness_change, fixation_population_size)
        return nothing
    end

    mutation_rate_multiplier_change = (
        if mutation_rate_multiplier_shape !== nothing
            Random.randexp(compartment.randomiser) / mutation_rate_multiplier_shape
        else
            0.0 # not using variable mutation rates
        end
    )

    if compartment.mutation_phylogeny !== nothing
        new_mutational_profile_id = MutationPhylogenyClass.add_mutation!(
            compartment.mutation_phylogeny,
            compartment.cell_phylogeny.mutational_profile_ids[cell_id],
            cell_id,
            fitness_change,
            smoking_signature,
            step,
        )
        CellPhylogenyClass.add_mutation!(
            compartment.cell_phylogeny,
            cell_id,
            new_mutational_profile_id,
            fitness_change,
            smoking_driver_fitness_augmentation,
            compartment.protection_coefficient,
            mutation_rate_multiplier_change,
        )
    end
    return fitness_change
end

function mutation_fixates(
    compartment::AbstractCompartment,
    fitness_change::Float64,
    quiescent_gland_size::Int,
)::Bool
    @assert compartment.cell_type == CellPhylogenyClass.quiescent
    fixation_probability = (
        if fitness_change == 0
            1 / quiescent_gland_size
        elseif fitness_change * quiescent_gland_size > 50
            # very good approximation in this case, removes numerical issues
            1 - exp(-fitness_change)
        else
            (
                # from Ewens, 1979
                exp(fitness_change * (quiescent_gland_size - 1)) *
                (exp(fitness_change) - 1) /
                (exp(fitness_change * quiescent_gland_size) - 1)
            )
        end
    )
    @assert 0 <= fixation_probability <= 1 "fitness_change $fitness_change, fixation_probability $fixation_probability"
    return rand(compartment.randomiser) < fixation_probability
end

function randomise_division_type(
    compartment::Compartment,
    cell_id::UInt64,
    symmetric_division_prob::Float64,
    smoking::Bool,
    rnd::Float64,
)::CellPhylogenyClass.DivisionType
    projected_cell_fitness = CurrentFitnessClass.get_projected_fitness(
        compartment[cell_id].current_fitness,
        symmetric_division_prob,
        smoking,
    )
    adjusted_symmetric_division_prob = symmetric_division_prob + projected_cell_fitness
    adjusted_symmetric_differentiation_prob =
        symmetric_division_prob - projected_cell_fitness
    if rnd < adjusted_symmetric_division_prob
        return CellPhylogenyClass.symmetric_division
    elseif (1 - rnd) < adjusted_symmetric_differentiation_prob
        return CellPhylogenyClass.symmetric_differentiation
    else
        return CellPhylogenyClass.asymmetric_division
    end
end

function get_dynamic_normalisation_adjustment(
    compartment::Compartment,
    dynamic_normalisation_power::Float64,
)::Float64
    return dynamic_normalisation_power *
           log(compartment.initial_cell_number / cell_count(compartment))
end

end
