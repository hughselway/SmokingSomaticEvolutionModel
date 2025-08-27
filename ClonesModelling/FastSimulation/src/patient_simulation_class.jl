module PatientSimulationClass

using Random
using LoggingExtras
using StatsBase

using ..CompartmentClass
using ..CellPopulationClass
using ..CellPhylogenyClass
using ..CondensedPhylogenyClass
using ..SmokingAffectedParameterClass
using ..SmokingRecordClass
using ..Records

mutable struct CellCountBounds
    min::Int
    max::Int
end

abstract type AbstractPatientSimulation end

struct PatientSimulation <: AbstractPatientSimulation
    smoking_record::SmokingRecordClass.SmokingRecord
    dynamic_normalisation_power::Float64
    steps_per_year::Int
    cell_population::CellPopulationClass.CellPopulation
    logger::Union{AbstractLogger,Nothing}
    mutational_burden_record::Union{Records.Record,Nothing}
    fitness_summaries::Union{Records.Record,Nothing}
    symmetric_division_prob::Float64
    expected_divisions_per_step::SmokingAffectedParameterClass.SmokingAffectedParameter
    expected_mutations_per_division::SmokingAffectedParameterClass.SmokingAffectedParameter
    fitness_change_scale::Float64
    mutation_driver_probability::Float64
    fitness_change_probability::Float64
    quiescent_module::Bool
    quiescent_protected_module::Bool
    quiescent_expected_divisions_per_step::SmokingAffectedParameterClass.SmokingAffectedParameter
    protected_module::Bool
    immune_response_module::Bool
    immune_death_prob_per_mutation_each_step::SmokingAffectedParameterClass.SmokingAffectedParameter
    smoking_driver_module::Bool
    smoking_driver_fitness_augmentation::Float64
    mutation_rate_multiplier_shape::Union{Float64,Nothing}
    cell_count_bounds::CellCountBounds
end

function PatientSimulation(
    smoking_record::SmokingRecordClass.SmokingRecord,
    randomiser::Random.Xoshiro,
    initial_main_population_size::Int,
    steps_per_year::Int,
    dynamic_normalisation_power::Float64,
    record_frequency::Int,
    fitness_change_scale::Float64,
    non_smoking_mutations_per_year::Float64,
    smoking_mutations_per_year::Float64,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    non_smoking_divisions_per_year::Float64,
    smoking_divisions_per_year::Float64,
    quiescent_main_population_size::Int,
    quiescent_divisions_per_year::Float64,
    quiescent_protection_coefficient::Float64,
    initial_protected_population_size::Int,
    protection_coefficient::Float64,
    quiescent_protected_population_size::Int,
    immune_death_rate::Float64,
    smoking_immune_coeff::Float64,
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    csv_logging::Bool,
    record_phylogenies::Bool,
    this_probe_logging_directory::Union{String,Nothing} = nothing,
    console_logging_level::Union{LoggingExtras.LogLevel,Nothing} = LoggingExtras.Info,
    file_logging_level::Union{LoggingExtras.LogLevel,Nothing} = LoggingExtras.Debug,
)::PatientSimulation
    if !record_phylogenies
        throw(
            NotImplementedError(
                "Only record_phylogenies = true is currently supported for non-spatial simulations",
            ),
        ) # TODO: implement
    end
    use_yearly_counters =
        console_logging_level !== nothing ||
        file_logging_level !== nothing ||
        csv_logging
    cell_population = CellPopulationClass.CellPopulation(
        randomiser,
        initial_main_population_size,
        quiescent_main_population_size,
        initial_protected_population_size,
        quiescent_protected_population_size,
        protection_coefficient,
        use_yearly_counters,
        mutation_rate_multiplier_shape !== nothing,
        record_phylogenies,
        expected_total_mutational_profiles(
            smoking_record,
            smoking_mutations_per_year,
            non_smoking_mutations_per_year,
            initial_main_population_size + initial_protected_population_size,
        ),
        expected_total_cells(
            smoking_record,
            smoking_divisions_per_year,
            non_smoking_divisions_per_year,
            initial_main_population_size + initial_protected_population_size,
        ),
        quiescent_protection_coefficient,
    )
    if use_yearly_counters
        @assert this_probe_logging_directory !== nothing
        mkpath(this_probe_logging_directory)
    end
    logger = (
        if console_logging_level === nothing && file_logging_level === nothing
            nothing
        else
            TeeLogger(
                MinLevelLogger(ConsoleLogger(), console_logging_level),
                MinLevelLogger(
                    FileLogger(
                        joinpath(
                            this_probe_logging_directory,
                            smoking_record.patient * ".log",
                        ),
                    ),
                    file_logging_level,
                ),
            )
        end
    )
    mutational_burden_record = (
        if record_frequency > 0
            Records.MutationalBurdenRecord(
                ceil(
                    Int,
                    (smoking_record.age + 1) *
                    record_frequency *
                    sum(
                        compartment.initial_cell_number for compartment in
                        values(cell_population.cell_compartments_dict)
                    ),
                ),
            )
        else
            nothing
        end
    )
    fitness_summaries = (
        if record_frequency > 0
            Records.FitnessRecord(
                smoking_record.age,
                record_frequency,
                length(cell_population.cell_compartments_dict),
            )
        else
            nothing
        end
    )
    symmetric_division_prob = 1.0 / (2 * non_smoking_divisions_per_year)
    symmetric_division_prob > 0.5 && error(
        "LowDivisionRateError: Symmetric division probability must be less than 0.5",
    )
    expected_divisions_per_step =
        SmokingAffectedParameterClass.SmokingAffectedParameterClass.SmokingAffectedParameter(
            smoking_divisions_per_year / steps_per_year,
            non_smoking_divisions_per_year / steps_per_year,
        )
    expected_mutations_per_division =
        SmokingAffectedParameterClass.SmokingAffectedParameterClass.SmokingAffectedParameter(
            smoking_mutations_per_year / smoking_divisions_per_year,
            non_smoking_mutations_per_year / non_smoking_divisions_per_year,
        )
    quiescent_module = quiescent_main_population_size > 0
    quiescent_protected_module = quiescent_protection_coefficient < 1
    if quiescent_module
        quiescent_divisions_per_step = (quiescent_divisions_per_year / steps_per_year)
        quiescent_expected_divisions_per_step =
            SmokingAffectedParameterClass.SmokingAffectedParameter(
                if quiescent_protected_module
                    (
                        (smoking_divisions_per_year / non_smoking_divisions_per_year) *
                        quiescent_divisions_per_step
                    )
                else
                    quiescent_divisions_per_step
                end,
                quiescent_divisions_per_step,
            )
    else
        quiescent_expected_divisions_per_step =
            (SmokingAffectedParameterClass.SmokingAffectedParameter(0.0, 0.0))
    end

    protected_module = initial_protected_population_size > 0
    immune_response_module = immune_death_rate > 0
    if immune_response_module
        immune_death_prob_per_mutation_each_step =
            SmokingAffectedParameterClass.SmokingAffectedParameter(
                immune_death_rate * smoking_immune_coeff / steps_per_year,
                immune_death_rate / steps_per_year,
            )
    else
        immune_death_prob_per_mutation_each_step =
            (SmokingAffectedParameterClass.SmokingAffectedParameter(0.0, 0.0))
    end

    smoking_driver_module = smoking_driver_fitness_augmentation > 0
    return PatientSimulation(
        smoking_record,
        dynamic_normalisation_power,
        steps_per_year,
        cell_population,
        logger,
        mutational_burden_record,
        fitness_summaries,
        symmetric_division_prob,
        expected_divisions_per_step,
        expected_mutations_per_division,
        fitness_change_scale,
        mutation_driver_probability,
        fitness_change_probability,
        quiescent_module,
        quiescent_protected_module,
        quiescent_expected_divisions_per_step,
        protected_module,
        immune_response_module,
        immune_death_prob_per_mutation_each_step,
        smoking_driver_module,
        smoking_driver_fitness_augmentation,
        mutation_rate_multiplier_shape,
        CellCountBounds(
            initial_main_population_size +
            initial_protected_population_size +
            quiescent_main_population_size +
            quiescent_protected_population_size,
            0,
        ),
    )
end

function run!(
    patient_simulation::PatientSimulation,
    record_frequency::Int,
    this_run_logging_directory::Union{String,Nothing},
    csv_logging::Bool,
)::Bool
    if csv_logging
        patient_simulation_record = Records.PatientSimulationRecord(
            ceil(Int, patient_simulation.smoking_record.age),
        )
    elseif patient_simulation.logger !== nothing
        initial_log(patient_simulation)
    end

    age_in_steps::Int =
        ceil(patient_simulation.smoking_record.age * patient_simulation.steps_per_year)
    zero_population_error = false
    for step_number in collect(UInt16, 1:age_in_steps)
        age = step_number / patient_simulation.steps_per_year
        smoking =
            SmokingRecordClass.smoking_at_age(patient_simulation.smoking_record, age)
        zero_population_error =
            increment_one_step!(patient_simulation, smoking, step_number)
        if record_frequency > 0 &&
           (
            (step_number % patient_simulation.steps_per_year) %
            ceil(patient_simulation.steps_per_year / record_frequency)
        ) == 0
            CellPopulationClass.record_mutational_burden(
                patient_simulation.cell_population,
                patient_simulation.mutational_burden_record,
                step_number,
            )
            CellPopulationClass.record_fitness_summary(
                patient_simulation.cell_population,
                patient_simulation.fitness_summaries,
                step_number,
            )
        end
        CellPopulationClass.increment_age_one_step!(patient_simulation.cell_population)
        if step_number % patient_simulation.steps_per_year == 0
            update_cell_count_bounds!(patient_simulation)

            year_number = Int(age)
            if csv_logging
                CellPopulationClass.add_yearly_record(
                    patient_simulation.cell_population,
                    patient_simulation_record,
                    year_number,
                )
            elseif patient_simulation.logger !== nothing
                log_year(patient_simulation, year_number)
            end

            if patient_simulation.cell_population.cell_compartments_dict["main"].yearly_counters !==
               nothing
                CellPopulationClass.reset_yearly_counters!(
                    patient_simulation.cell_population,
                )
            end
        end

        if zero_population_error
            break
        end
    end
    if record_frequency > 0
        Records.write_mutational_burden_record(
            patient_simulation.mutational_burden_record,
            this_run_logging_directory,
            patient_simulation.smoking_record.patient,
            true,
        )
        Records.write_fitness_summaries(
            patient_simulation.fitness_summaries,
            this_run_logging_directory,
            patient_simulation.smoking_record.patient,
        )
    end
    if csv_logging
        Records.write_to_csv(
            patient_simulation_record,
            joinpath(
                this_run_logging_directory,
                "$(patient_simulation.smoking_record.patient).csv",
            ),
        )
    elseif patient_simulation.logger !== nothing
        with_logger(patient_simulation.logger) do
            if !zero_population_error
                @info "Simulation complete"
            end
        end
    end
    return zero_population_error
end

function initial_log(patient_simulation::PatientSimulation)::Nothing
    with_logger(patient_simulation.logger) do
        @info "Running simulation for patient $(patient_simulation.smoking_record.patient)"
        @debug (
            "Smoking record $(patient_simulation.smoking_record)\n" *
            "Dynamic normalisation power $(patient_simulation.dynamic_normalisation_power)\n" *
            "Steps per year $(patient_simulation.steps_per_year)\n" *
            "$(string(patient_simulation.cell_population))\n" *
            "Symmetric division probability $(patient_simulation.symmetric_division_prob)\n" *
            "Division probability each step $(patient_simulation.expected_divisions_per_step)\n" *
            "Mutation probability on division $(patient_simulation.expected_mutations_per_division)\n" *
            "Fitness change scale $(patient_simulation.fitness_change_scale)\n" *
            "Mutation driver probability $(patient_simulation.mutation_driver_probability)\n" *
            "Quiescent module $(patient_simulation.quiescent_module)\n" *
            "Quiescent protected module $(patient_simulation.quiescent_protected_module)\n" *
            "Quiescent division probability each step $(patient_simulation.quiescent_expected_divisions_per_step)\n" *
            "Protected module $(patient_simulation.protected_module)\n" *
            "Immune response module $(patient_simulation.immune_response_module)\n" *
            "Immune death probability per mutation each step $(patient_simulation.immune_death_prob_per_mutation_each_step)\n" *
            "Smoking driver module $(patient_simulation.smoking_driver_module)\n" *
            "Smoking driver fitness augmentation $(patient_simulation.smoking_driver_fitness_augmentation)"
        )
    end
    return nothing
end

function increment_one_step!(
    patient_simulation::PatientSimulation,
    smoking::Bool,
    step::UInt16,
)::Bool
    CellPopulationClass.normalise_fitness!(
        patient_simulation.cell_population,
        patient_simulation.dynamic_normalisation_power,
        smoking,
    )

    for compartment in values(patient_simulation.cell_population.cell_compartments_dict)
        new_cell_ids = CompartmentClass.divide_cells!(
            compartment,
            smoking,
            (
                if (compartment.cell_type == CellPhylogenyClass.basal)
                    patient_simulation.expected_divisions_per_step
                else
                    patient_simulation.quiescent_expected_divisions_per_step
                end
            ),
            patient_simulation.expected_mutations_per_division,
            patient_simulation.smoking_driver_fitness_augmentation,
            patient_simulation.mutation_rate_multiplier_shape,
            patient_simulation.symmetric_division_prob,
            patient_simulation.mutation_driver_probability,
            patient_simulation.fitness_change_probability,
            patient_simulation.fitness_change_scale,
            step,
        )
        if length(new_cell_ids) > 0
            @assert compartment.destination_compartment_name !== nothing
            append!(
                patient_simulation.cell_population.cell_compartments_dict[compartment.destination_compartment_name].alive_cell_ids,
                new_cell_ids,
            )
        end
        if patient_simulation.immune_response_module
            CompartmentClass.apply_immune_system!(
                compartment,
                SmokingAffectedParameterClass.get_value(
                    patient_simulation.immune_death_prob_per_mutation_each_step,
                    smoking,
                ),
                step,
            )
        end
        if compartment.cell_type == CellPhylogenyClass.basal &&
           length(compartment.alive_cell_ids) == 0
            if patient_simulation.logger !== nothing
                with_logger(patient_simulation.logger) do
                    @error "Basal compartment empty at step $(step)"
                end
            end
            return true
        end
    end
    return false
end

function log_year(patient_simulation::PatientSimulation, year_number::Integer)::Nothing
    with_logger(patient_simulation.logger) do
        @debug (
            "Completed $year_number years\n\t" * join(
                [
                    compartment_name *
                    ": " *
                    CompartmentClass.yearly_stats_string(compartment) for
                    (compartment_name, compartment) in
                    pairs(patient_simulation.cell_population.cell_compartments_dict)
                ],
                "\n\t",
            )
        )
    end
    return nothing
end

function update_cell_count_bounds!(patient_simulation::PatientSimulation)::Nothing
    cell_count =
        CellPopulationClass.number_of_basal_cells(patient_simulation.cell_population)
    if cell_count < patient_simulation.cell_count_bounds.min
        patient_simulation.cell_count_bounds.min = cell_count
    end
    if cell_count > patient_simulation.cell_count_bounds.max
        patient_simulation.cell_count_bounds.max = cell_count
    end
    return nothing
end

function expected_total_mutational_profiles(
    smoking_record::SmokingRecordClass.SmokingRecord,
    smoking_mutations_per_year::Float64,
    non_smoking_mutations_per_year::Float64,
    initial_basal_cell_number::Int,
)::Int
    expected_total_mutational_profiles = ceil(
        Int,
        initial_basal_cell_number * (
            non_smoking_mutations_per_year *
            SmokingRecordClass.years_not_smoking(smoking_record) +
            smoking_mutations_per_year *
            SmokingRecordClass.years_smoking(smoking_record)
        ),
    )
    if expected_total_mutational_profiles == 0
        throw(
            ArgumentError(
                "expected total mutation profiles is zero; initial_basal_cell_number = $initial_basal_cell_number, non_smoking_mutations_per_year = $non_smoking_mutations_per_year, smoking_mutations_per_year = $smoking_mutations_per_year, smoking_record = $smoking_record",
            ),
        )
    end
    return expected_total_mutational_profiles
end

function expected_total_cells(
    smoking_record::SmokingRecordClass.SmokingRecord,
    smoking_divisions_per_year::Float64,
    non_smoking_divisions_per_year::Float64,
    initial_basal_cell_number::Int,
)::Int
    return ceil(
        Int,
        smoking_record.age *
        initial_basal_cell_number *
        (
            if smoking_record.status == "never"
                non_smoking_divisions_per_year
            else
                smoking_divisions_per_year
            end
        ),
    )
end

function get_final_mutational_burden(
    patient_simulation::AbstractPatientSimulation,
)::Tuple{Vector{Int},Vector{Int}}
    return CellPopulationClass.get_final_mutational_burden(
        patient_simulation.cell_population,
    )
end

function get_cell_population_subsets(
    alive_cell_ids::Vector{UInt64},
    alive_cell_weights::AbstractWeights{Int},
    subsample_count::Union{Int,Nothing},
    subsample_size::Union{UInt16,Nothing},
    randomiser::Random.Xoshiro,
    spatial::Bool,
)::Union{Vector{Nothing},Vector{Vector{UInt64}}}
    if subsample_count === nothing
        return [nothing]
    end
    if subsample_size === nothing
        subsample_size = 50
    end
    if subsample_size >= length(alive_cell_ids)
        @assert !spatial
        return [alive_cell_ids for _ in 1:subsample_count]
    end
    if typeof(alive_cell_weights) == StatsBase.UnitWeights{Int}
        return [
            StatsBase.sample(
                randomiser,
                alive_cell_ids,
                subsample_size,
                replace = false,
            ) for _ in 1:subsample_count
        ]
    end
    return [
        StatsBase.sample(
            randomiser,
            alive_cell_ids,
            alive_cell_weights,
            subsample_size,
            replace = false,
        ) for _ in 1:subsample_count
    ]
end

end
