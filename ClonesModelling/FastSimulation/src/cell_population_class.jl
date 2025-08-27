module CellPopulationClass

using Random
import Base: string
using Statistics
using OrderedCollections: OrderedDict

using ..CellPhylogenyClass
using ..CompartmentClass
using ..MutationPhylogenyClass
using ..Records
using ..CurrentFitnessClass

abstract type AbstractCellPopulation end

struct CellPopulation <: AbstractCellPopulation
    randomiser::Random.Xoshiro
    initial_main_population_size::Int
    quiescent_main_population_size::Int
    initial_protected_population_size::Int
    quiescent_protected_population_size::Int
    protection_coefficient::Float64
    quiescent_protection_coefficient::Float64
    current_age_in_steps::Ref{Int}
    cell_compartments_dict::OrderedDict{String,CompartmentClass.Compartment}
end

function CellPopulation(
    randomiser::Random.Xoshiro,
    initial_main_population_size::Int,
    quiescent_main_population_size::Int,
    initial_protected_population_size::Int,
    quiescent_protected_population_size::Int,
    protection_coefficient::Float64,
    use_yearly_counters::Bool,
    vary_mutation_rate::Bool,
    record_phylogenies::Bool,
    expected_total_mutational_profiles::Int,
    expected_total_cells::Int,
    quiescent_protection_coefficient::Float64 = nothing,
)::CellPopulation
    if record_phylogenies
        mutation_phylogeny =
            MutationPhylogenyClass.MutationPhylogeny(expected_total_mutational_profiles)
        cell_phylogeny = CellPhylogenyClass.CellPhylogeny(expected_total_cells)
    else
        mutation_phylogeny = nothing
        cell_phylogeny = CellPhylogenyClass.EmptyCellPhylogeny()
    end

    cell_compartments_dict = OrderedDict(
        "main" => CompartmentClass.Compartment(
            randomiser,
            mutation_phylogeny,
            cell_phylogeny,
            CellPhylogenyClass.basal,
            initial_main_population_size,
            1.0,
            use_yearly_counters,
            true,
            true,
            true,
            vary_mutation_rate,
        ),
    )

    if quiescent_main_population_size > 0
        @assert quiescent_protection_coefficient !== nothing
        cell_compartments_dict["quiescent_main"] = CompartmentClass.Compartment(
            randomiser,
            mutation_phylogeny,
            cell_phylogeny,
            CellPhylogenyClass.quiescent,
            quiescent_main_population_size,
            quiescent_protection_coefficient,
            use_yearly_counters,
            false,
            false,
            false,
            vary_mutation_rate,
            "main",
        )
    end

    if initial_protected_population_size > 0
        cell_compartments_dict["protected"] = CompartmentClass.Compartment(
            randomiser,
            mutation_phylogeny,
            cell_phylogeny,
            CellPhylogenyClass.basal,
            initial_protected_population_size,
            protection_coefficient,
            use_yearly_counters,
            true,
            true,
            true,
            vary_mutation_rate,
        )
    end

    if quiescent_protected_population_size > 0
        cell_compartments_dict["quiescent_protected"] = CompartmentClass.Compartment(
            randomiser,
            mutation_phylogeny,
            cell_phylogeny,
            CellPhylogenyClass.quiescent,
            quiescent_protected_population_size,
            protection_coefficient,
            use_yearly_counters,
            false,
            false,
            false,
            vary_mutation_rate,
            "protected",
        )
    end
    return CellPopulation(
        randomiser,
        initial_main_population_size,
        quiescent_main_population_size,
        initial_protected_population_size,
        quiescent_protected_population_size,
        protection_coefficient,
        quiescent_protection_coefficient,
        Ref(0),
        cell_compartments_dict,
    )
end

function string(cell_population::CellPopulation)::String
    if CompartmentClass.cell_count(cell_population.cell_compartments_dict["main"]) == 0
        return "ZeroPopulationError: The main population has zero cells"
    elseif (
        cell_population.initial_protected_population_size > 0 &&
        CompartmentClass.cell_count(
            cell_population.cell_compartments_dict["protected"],
        ) == 0
    )
        return "ZeroPopulationError: The protected population has zero cells"
    end

    compartments_summary::String = join(
        [
            "$compartment_name compartment: " *
            string(
                CompartmentClass.cell_count(
                    cell_population.cell_compartments_dict[compartment_name],
                ),
            ) *
            " cells, with mean mutation counts: " *
            join(
                [
                    string(
                        mean(
                            CompartmentClass.get_mutation_counts_list(
                                compartment,
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
            pairs(cell_population.cell_compartments_dict)
        ],
        "\n\t",
    )
    return "Cell population:\n\t$compartments_summary"
end

function number_of_basal_cells(cell_population::CellPopulation)::Int
    return sum(
        CompartmentClass.cell_count(compartment) for (compartment_name, compartment) in
        pairs(cell_population.cell_compartments_dict) if
        compartment.cell_type == CellPhylogenyClass.basal
    )
end

function normalise_fitness!(
    cell_population::CellPopulation,
    dynamic_normalisation_power::Float64,
    smoking::Bool,
)::Nothing
    @assert CompartmentClass.cell_count(
        cell_population.cell_compartments_dict["main"],
    ) > 0
    if cell_population.initial_protected_population_size > 0
        @assert CompartmentClass.cell_count(
            cell_population.cell_compartments_dict["protected"],
        ) > 0
    end

    mean_fitness = get_mean_fitness(cell_population, smoking)
    for compartment in values(cell_population.cell_compartments_dict)
        if compartment.cell_type == CellPhylogenyClass.basal &&
           compartment.included_in_competition
            value_to_subtract =
                mean_fitness - CompartmentClass.get_dynamic_normalisation_adjustment(
                    compartment,
                    dynamic_normalisation_power,
                )
            for cell_id in compartment.alive_cell_ids
                CurrentFitnessClass.subtract!(
                    compartment.cell_phylogeny[cell_id].current_fitness,
                    value_to_subtract,
                )
            end
        end
    end
    return nothing
end

function get_mean_fitness(cell_population::CellPopulation, smoking::Bool)::Float64
    # TODO keep track of this in the cell population, change on mutation
    return sum(
        CompartmentClass.get_total_fitness(compartment, smoking) for
        compartment in values(cell_population.cell_compartments_dict) if
        compartment.included_in_competition
    ) / (sum(
        CompartmentClass.cell_count(compartment) for
        compartment in values(cell_population.cell_compartments_dict) if
        compartment.included_in_competition
    ))
end

function reset_yearly_counters!(cell_population::AbstractCellPopulation)::Nothing
    for compartment in values(cell_population.cell_compartments_dict)
        CompartmentClass.reset!(compartment.yearly_counters)
    end
    return nothing
end

function add_yearly_record(
    cell_population::AbstractCellPopulation,
    patient_simulation_record::Records.Record,
    year::Integer,
)::Nothing
    for (compartment_name, compartment) in pairs(cell_population.cell_compartments_dict)
        Records.add_yearly_record!(
            patient_simulation_record,
            UInt8(year),
            compartment_name,
            CompartmentClass.cell_count(compartment),
            compartment.yearly_counters.this_year_new_cell_count,
            compartment.yearly_counters.this_year_differentiate_count,
            compartment.yearly_counters.this_year_immune_death_count,
        )
    end
end

function record_mutational_burden(
    cell_population::AbstractCellPopulation,
    mutational_burden_record::Records.Record,
    step_number::UInt16,
)::Nothing
    for (compartment_name, compartment) in pairs(cell_population.cell_compartments_dict)
        if compartment.to_be_counted
            CompartmentClass.record_mutational_burden(
                compartment,
                mutational_burden_record,
                step_number,
                compartment_name,
            )
        end
    end
    return nothing
end

function get_final_mutational_burden(
    cell_population::CellPopulation,
)::Tuple{Vector{Int},Vector{Int}}
    final_mutational_burden = Vector{Int}(undef, 0)
    final_smoking_status_mutational_burden = Vector{Int}(undef, 0)
    for compartment in values(cell_population.cell_compartments_dict)
        if compartment.to_be_counted
            for cell_id in compartment.alive_cell_ids
                mutational_profile_id =
                    compartment.cell_phylogeny.mutational_profile_ids[cell_id]
                push!(
                    final_mutational_burden,
                    MutationPhylogenyClass.mutation_count(
                        compartment.mutation_phylogeny,
                        mutational_profile_id,
                        nothing,
                        nothing,
                    ),
                )
                push!(
                    final_smoking_status_mutational_burden,
                    MutationPhylogenyClass.mutation_count(
                        compartment.mutation_phylogeny,
                        mutational_profile_id,
                        nothing,
                        true,
                    ),
                )
            end
        end
    end
    return final_mutational_burden, final_smoking_status_mutational_burden
end

function alive_cell_ids(cell_population::AbstractCellPopulation)::Vector{UInt64}
    return [
        cell_id for compartment in values(cell_population.cell_compartments_dict) for
        cell_id in compartment.alive_cell_ids
    ]
end

function record_fitness_summary(
    cell_population::CellPopulation,
    fitness_summaries::Records.Record,
    step_number::UInt16,
)::Nothing
    for (compartment_name, compartment) in pairs(cell_population.cell_compartments_dict)
        smoking_fitnesses = [
            CurrentFitnessClass.get_value(cell.current_fitness, true) for
            cell in CompartmentClass.cell_list(compartment)
        ]
        non_smoking_fitnesses = [
            CurrentFitnessClass.get_value(cell.current_fitness, false) for
            cell in CompartmentClass.cell_list(compartment)
        ]
        Records.record_fitness_summary(
            fitness_summaries,
            step_number,
            compartment_name,
            mean(smoking_fitnesses),
            mean(non_smoking_fitnesses),
            std(smoking_fitnesses),
            std(non_smoking_fitnesses),
            0.0,
        )
    end
    smoking_fitnesses = [
        CurrentFitnessClass.get_value(cell.current_fitness, true) for
        compartment in values(cell_population.cell_compartments_dict) for
        cell in CompartmentClass.cell_list(compartment)
    ]
    non_smoking_fitnesses = [
        CurrentFitnessClass.get_value(cell.current_fitness, false) for
        compartment in values(cell_population.cell_compartments_dict) for
        cell in CompartmentClass.cell_list(compartment)
    ]
    Records.record_fitness_summary(
        fitness_summaries,
        step_number,
        "total",
        mean(smoking_fitnesses),
        mean(non_smoking_fitnesses),
        std(smoking_fitnesses),
        std(non_smoking_fitnesses),
        0.0,
    )
    return nothing
end

function increment_age_one_step!(cell_population::CellPopulation)::Nothing
    cell_population.current_age_in_steps[] += 1
    return nothing
end

function get_mutation_counts_list(
    cell_population::CellPopulation,
    driver::Union{Bool,Nothing},
    smoking_signature::Union{Bool,Nothing},
)::Vector{Int}
    return vcat(
        [
            CompartmentClass.get_mutation_counts_list(
                compartment,
                driver,
                smoking_signature,
            ) for compartment in values(cell_population.cell_compartments_dict) if
            compartment.to_be_counted
        ]...,
    )
end

end
