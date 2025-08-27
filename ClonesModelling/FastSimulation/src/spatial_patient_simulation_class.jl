module SpatialPatientSimulationClass

using Random
using LoggingExtras
using DataFrames: DataFrame
using CSV
using Statistics

using ..CellLatticeClass
using ..CellPopulationClass
using ..CurrentFitnessClass
using ..PatientSimulationClass
using ..ParameterValuesClass
using ..SpatialCellPopulationClass
using ..Records
using ..SmokingAffectedParameterClass
using ..SmokingRecordClass
using ..SpatialCompartmentClass

struct SpatialPatientSimulation <: PatientSimulationClass.AbstractPatientSimulation
    smoking_record::SmokingRecordClass.SmokingRecord
    cell_population::SpatialCellPopulationClass.SpatialCellPopulation
    logger::Union{AbstractLogger,Nothing}
    mutational_burden_record::Union{Records.Record,Nothing}
    fitness_summaries::Union{Records.Record,Nothing}
    allow_early_stopping::Bool
    symmetric_division_prob::Float64
    divisions_per_year::SmokingAffectedParameterClass.SmokingAffectedParameter
    expected_mutations_per_division::SmokingAffectedParameterClass.SmokingAffectedParameter
    fitness_change_scale::Float64
    mutation_driver_probability::Float64
    fitness_change_probability::Float64
    quiescent_module::Bool
    quiescent_protected_module::Bool
    quiescent_division_rate_coeff::SmokingAffectedParameterClass.SmokingAffectedParameter
    ambient_quiescent_division_rate_coeff::SmokingAffectedParameterClass.SmokingAffectedParameter
    protected_module::Bool
    immune_response_module::Bool
    immune_death_rate::SmokingAffectedParameterClass.SmokingAffectedParameter
    smoking_driver_module::Bool
    smoking_driver_fitness_augmentation::Float64
    mutation_rate_multiplier_shape::Union{Float64,Nothing}
end

function SpatialPatientSimulation(
    smoking_record::SmokingRecordClass.SmokingRecord,
    randomiser::Random.Xoshiro,
    grid_side_length::Int,
    record_frequency::Int,
    quiescent_spacing::Int,
    protected_spacing::Int,
    parameter_values::ParameterValuesClass.ParameterValues,
    csv_logging::Bool,
    allow_early_stopping::Bool,
    record_phylogenies::Bool,
    this_probe_logging_directory::Union{String,Nothing},
    replicate_number::Int,
    console_logging_level::Union{Int,Nothing},
    file_logging_level::Union{Int,Nothing},
    annual_turnover::Float64 = 1.0, # from Teixeira et al. 2013
)::SpatialPatientSimulation
    return SpatialPatientSimulation(
        smoking_record,
        randomiser,
        grid_side_length,
        record_frequency,
        parameter_values.fitness_change_scale,
        parameter_values.non_smoking_mutations_per_year,
        parameter_values.smoking_mutations_per_year,
        ParameterValuesClass.get_mutation_driver_probability(
            parameter_values.fitness_change_scale,
            parameter_values.fitness_change_probability,
            true,
        ),
        parameter_values.fitness_change_probability,
        ParameterValuesClass.non_smoking_divisions_per_year(parameter_values),
        ParameterValuesClass.smoking_divisions_per_year(parameter_values),
        quiescent_spacing,
        if ParameterValuesClass.is_active(parameter_values, :Quiescent)
            parameter_values.quiescent_parameters.quiescent_divisions_per_year
        else
            0.0
        end,
        if ParameterValuesClass.is_active(parameter_values, :QuiescentProtected)
            parameter_values.quiescent_protected_parameters.quiescent_protection_coefficient
        else
            0.0
        end,
        if ParameterValuesClass.is_active(parameter_values, :Quiescent)
            parameter_values.quiescent_parameters.ambient_quiescent_divisions_per_year
        else
            0.0
        end,
        if ParameterValuesClass.is_active(parameter_values, :Quiescent)
            parameter_values.quiescent_parameters.quiescent_gland_cell_count
        else
            1
        end,
        protected_spacing,
        if ParameterValuesClass.is_active(parameter_values, :Protected)
            parameter_values.protected_parameters.protected_region_radius
        else
            0
        end,
        if ParameterValuesClass.is_active(parameter_values, :Protected)
            parameter_values.protected_parameters.protection_coefficient
        else
            0.0
        end,
        # quiescent_protected_population_size,
        if ParameterValuesClass.is_active(parameter_values, :ImmuneResponse)
            parameter_values.immune_response_parameters.immune_death_rate
        else
            0.0
        end,
        if ParameterValuesClass.is_active(parameter_values, :ImmuneResponse)
            parameter_values.immune_response_parameters.smoking_immune_coeff
        else
            0.0
        end,
        if ParameterValuesClass.is_active(parameter_values, :SmokingDriver)
            parameter_values.smoking_driver_parameters.smoking_driver_fitness_augmentation
        else
            0.0
        end,
        parameter_values.mutation_rate_multiplier_shape,
        csv_logging,
        allow_early_stopping,
        record_phylogenies,
        annual_turnover,
        if this_probe_logging_directory !== nothing
            joinpath(this_probe_logging_directory, "replicate_$(replicate_number-1)")
        else
            nothing
        end,
        if console_logging_level !== nothing
            Logging.LogLevel(console_logging_level)
        else
            nothing
        end,
        if file_logging_level !== nothing
            Logging.LogLevel(file_logging_level)
        else
            nothing
        end,
    )
end

function SpatialPatientSimulation(
    smoking_record::SmokingRecordClass.SmokingRecord,
    randomiser::Random.Xoshiro,
    grid_side_length::Int,
    record_frequency::Int,
    fitness_change_scale::Float64,
    non_smoking_mutations_per_year::Float64,
    smoking_mutations_per_year::Float64,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
    non_smoking_divisions_per_year::Float64,
    smoking_divisions_per_year::Float64,
    quiescent_spacing::Int,
    quiescent_divisions_per_year::Float64,
    quiescent_protection_coefficient::Float64,
    ambient_quiescent_divisions_per_year::Float64,
    quiescent_gland_cell_count::Int,
    protected_spacing::Int,
    protected_region_radius::Union{Int,Nothing},
    protection_coefficient::Float64,
    immune_death_rate::Float64,
    smoking_immune_coeff::Float64,
    smoking_driver_fitness_augmentation::Float64,
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    csv_logging::Bool,
    allow_early_stopping::Bool,
    record_phylogenies::Bool,
    annual_turnover::Float64 = 1.0, # from Teixeira et al. 2013
    this_probe_logging_directory::Union{String,Nothing} = nothing,
    console_logging_level::Union{LoggingExtras.LogLevel,Nothing} = LoggingExtras.Info,
    file_logging_level::Union{LoggingExtras.LogLevel,Nothing} = LoggingExtras.Debug,
)::SpatialPatientSimulation
    any_logs =
        console_logging_level !== nothing ||
        file_logging_level !== nothing ||
        csv_logging
    use_yearly_counters = any_logs || allow_early_stopping

    cell_population = SpatialCellPopulationClass.SpatialCellPopulation(
        randomiser,
        grid_side_length,
        quiescent_spacing,
        protected_spacing,
        protected_region_radius,
        quiescent_gland_cell_count,
        protection_coefficient,
        quiescent_protection_coefficient,
        use_yearly_counters,
        smoking_driver_fitness_augmentation,
        mutation_rate_multiplier_shape !== nothing,
        record_phylogenies,
        PatientSimulationClass.expected_total_mutational_profiles(
            smoking_record,
            smoking_mutations_per_year,
            non_smoking_mutations_per_year,
            grid_side_length^2,
        ),
        PatientSimulationClass.expected_total_cells(
            smoking_record,
            smoking_divisions_per_year,
            non_smoking_divisions_per_year,
            grid_side_length^2,
        ),
    )
    if any_logs
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
            Records.SpatialMutationalBurdenRecord(
                ceil(
                    Int,
                    (smoking_record.age + 1) *
                    record_frequency *
                    sum(
                        compartment.initial_cell_number for compartment in
                        values(cell_population.cell_compartments_dict) if
                        compartment.to_be_counted
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
    symmetric_division_prob = annual_turnover / (2 * non_smoking_divisions_per_year)
    symmetric_division_prob > 0.5 && error(
        "LowDivisionRateError: Symmetric division probability must be less than 0.5",
    )
    divisions_per_year =
        SmokingAffectedParameterClass.SmokingAffectedParameterClass.SmokingAffectedParameter(
            smoking_divisions_per_year,
            non_smoking_divisions_per_year,
        )
    expected_mutations_per_division =
        SmokingAffectedParameterClass.SmokingAffectedParameterClass.SmokingAffectedParameter(
            smoking_mutations_per_year / smoking_divisions_per_year,
            non_smoking_mutations_per_year / non_smoking_divisions_per_year,
        )
    quiescent_module = quiescent_spacing > 0
    quiescent_protected_module = quiescent_protection_coefficient < 1
    if quiescent_module
        quiescent_division_rate_coeff =
            SmokingAffectedParameterClass.SmokingAffectedParameter(
                # multiply by 4 because only they have 1/4 as many neighbours
                4 * (
                    (smoking_divisions_per_year / non_smoking_divisions_per_year) *
                    quiescent_divisions_per_year
                ) / divisions_per_year.non_smoking_value,
                4 * quiescent_divisions_per_year / divisions_per_year.non_smoking_value,
            )
        ambient_quiescent_division_rate_coeff =
            SmokingAffectedParameterClass.SmokingAffectedParameter(
                if quiescent_protected_module
                    (ambient_quiescent_divisions_per_year) /
                    divisions_per_year.smoking_value
                else
                    ambient_quiescent_divisions_per_year /
                    divisions_per_year.non_smoking_value
                end,
                ambient_quiescent_divisions_per_year /
                divisions_per_year.non_smoking_value,
            )
    else
        quiescent_division_rate_coeff =
            (SmokingAffectedParameterClass.SmokingAffectedParameter(0.0, 0.0))
        ambient_quiescent_division_rate_coeff =
            (SmokingAffectedParameterClass.SmokingAffectedParameter(0.0, 0.0))
    end

    protected_module = protected_spacing > 0
    immune_response_module = immune_death_rate > 0
    if immune_response_module
        immune_death_rate = SmokingAffectedParameterClass.SmokingAffectedParameter(
            immune_death_rate * smoking_immune_coeff,
            immune_death_rate,
        )
    else
        immune_death_rate =
            SmokingAffectedParameterClass.SmokingAffectedParameter(0.0, 0.0)
    end

    smoking_driver_module = smoking_driver_fitness_augmentation > 0
    return SpatialPatientSimulation(
        smoking_record,
        cell_population,
        logger,
        mutational_burden_record,
        fitness_summaries,
        allow_early_stopping,
        symmetric_division_prob,
        divisions_per_year,
        expected_mutations_per_division,
        fitness_change_scale,
        mutation_driver_probability,
        fitness_change_probability,
        quiescent_module,
        quiescent_protected_module,
        quiescent_division_rate_coeff,
        ambient_quiescent_division_rate_coeff,
        protected_module,
        immune_response_module,
        immune_death_rate,
        smoking_driver_module,
        smoking_driver_fitness_augmentation,
        mutation_rate_multiplier_shape,
    )
end

function run!(
    spatial_patient_simulation::SpatialPatientSimulation,
    record_frequency::Int,
    this_run_logging_directory::Union{String,Nothing},
    csv_logging::Bool,
)::Nothing
    spatial_patient_simulation_record = nothing
    if csv_logging
        spatial_patient_simulation_record = Records.PatientSimulationRecord(
            ceil(Int, spatial_patient_simulation.smoking_record.age),
        )
    elseif spatial_patient_simulation.logger !== nothing
        initial_log(spatial_patient_simulation)
    end

    current_age::Float64 = 0.0
    current_year::Ref{UInt8} = Ref(UInt8(0))
    current_record_number::Ref{UInt16} = Ref(UInt16(0))
    smoking = false

    # so this doesn't need to be calculated at every timestep
    expected_mutations_per_division::Dict{Bool,Dict{Bool,Float64}} = Dict(
        true => Dict(
            true => SmokingAffectedParameterClass.get_protected_value(
                spatial_patient_simulation.expected_mutations_per_division,
                true,
                spatial_patient_simulation.cell_population.protection_coefficient,
            ),
            false =>
                spatial_patient_simulation.expected_mutations_per_division.non_smoking_value,
        ),
        false => Dict(
            true =>
                spatial_patient_simulation.expected_mutations_per_division.smoking_value,
            false =>
                spatial_patient_simulation.expected_mutations_per_division.non_smoking_value,
        ),
    )

    smoking = false
    while true
        time_to_next_event, event_type = SpatialCellPopulationClass.select_next_event(
            spatial_patient_simulation.cell_population,
            smoking,
            spatial_patient_simulation.divisions_per_year,
            spatial_patient_simulation.symmetric_division_prob,
            spatial_patient_simulation.immune_death_rate,
            spatial_patient_simulation.ambient_quiescent_division_rate_coeff,
        )
        current_age += time_to_next_event
        if current_age > spatial_patient_simulation.smoking_record.age
            break
        end
        smoking = SmokingRecordClass.smoking_at_age(
            spatial_patient_simulation.smoking_record,
            current_age,
        )
        if smoking != SmokingRecordClass.smoking_at_age(
            spatial_patient_simulation.smoking_record,
            current_age - time_to_next_event,
        )
            SpatialCellPopulationClass.recalibrate_current_normalisation_constant!(
                spatial_patient_simulation.cell_population,
                smoking,
            )
        end

        log_step!(
            spatial_patient_simulation,
            spatial_patient_simulation_record,
            current_age,
            current_year,
            current_record_number,
            csv_logging,
            record_frequency,
            smoking,
        )
        event_cell_location = SpatialCellPopulationClass.select_cell(
            spatial_patient_simulation.cell_population,
            smoking,
            event_type,
            spatial_patient_simulation.divisions_per_year,
            spatial_patient_simulation.symmetric_division_prob,
        )
        if event_type == SpatialCellPopulationClass.asymmetric_division
            SpatialCellPopulationClass.asymmetric_divide_cell!(
                spatial_patient_simulation.cell_population,
                event_cell_location,
                smoking,
                spatial_patient_simulation.mutation_driver_probability,
                spatial_patient_simulation.fitness_change_probability,
                spatial_patient_simulation.fitness_change_scale,
                expected_mutations_per_division,
                spatial_patient_simulation.smoking_driver_fitness_augmentation,
                spatial_patient_simulation.mutation_rate_multiplier_shape,
                current_record_number[],
            )
        elseif event_type == SpatialCellPopulationClass.symmetric_differentiation ||
               event_type == SpatialCellPopulationClass.immune_death
            SpatialCellPopulationClass.replace_cell!(
                spatial_patient_simulation.cell_population,
                event_cell_location,
                smoking,
                spatial_patient_simulation.divisions_per_year,
                spatial_patient_simulation.mutation_driver_probability,
                spatial_patient_simulation.fitness_change_probability,
                spatial_patient_simulation.symmetric_division_prob,
                spatial_patient_simulation.fitness_change_scale,
                expected_mutations_per_division,
                spatial_patient_simulation.quiescent_division_rate_coeff,
                spatial_patient_simulation.smoking_driver_fitness_augmentation,
                spatial_patient_simulation.mutation_rate_multiplier_shape,
                current_record_number[],
                event_type,
            )
        elseif event_type == SpatialCellPopulationClass.ambient_quiescent_division
            SpatialCellPopulationClass.ambient_quiescent_divide_cell!(
                spatial_patient_simulation.cell_population,
                event_cell_location,
                smoking,
                spatial_patient_simulation.mutation_driver_probability,
                spatial_patient_simulation.fitness_change_probability,
                spatial_patient_simulation.fitness_change_scale,
                expected_mutations_per_division,
                spatial_patient_simulation.smoking_driver_fitness_augmentation,
                spatial_patient_simulation.mutation_rate_multiplier_shape,
                current_record_number[],
            )
        end
    end

    if record_frequency > 0 && this_run_logging_directory !== nothing
        Records.write_mutational_burden_record(
            spatial_patient_simulation.mutational_burden_record,
            this_run_logging_directory,
            spatial_patient_simulation.smoking_record.patient,
            true,
        )
        Records.write_fitness_summaries(
            spatial_patient_simulation.fitness_summaries,
            this_run_logging_directory,
            spatial_patient_simulation.smoking_record.patient,
        )
    end

    if csv_logging
        Records.write_to_csv(
            spatial_patient_simulation_record,
            "$this_run_logging_directory/" *
            "$(spatial_patient_simulation.smoking_record.patient).csv",
        )
    end

    if spatial_patient_simulation.logger !== nothing
        with_logger(spatial_patient_simulation.logger) do
            @info "Patient $(spatial_patient_simulation.smoking_record.patient) simulation complete"
        end
    end
end

function initial_log(spatial_patient_simulation::SpatialPatientSimulation)::Nothing
    with_logger(spatial_patient_simulation.logger) do
        @info "Running simulation for patient $(spatial_patient_simulation.smoking_record.patient)"
        @debug (
            "Smoking record $(spatial_patient_simulation.smoking_record)\n" *
            "$(string(spatial_patient_simulation.cell_population))\n" *
            "Symmetric division probability $(spatial_patient_simulation.symmetric_division_prob)\n" *
            "Divisions per year $(spatial_patient_simulation.divisions_per_year)\n" *
            "Mutation probability on division $(spatial_patient_simulation.expected_mutations_per_division)\n" *
            "Fitness change scale $(spatial_patient_simulation.fitness_change_scale)\n" *
            "Mutation driver probability $(spatial_patient_simulation.mutation_driver_probability)\n" *
            "Quiescent module $(spatial_patient_simulation.quiescent_module)\n" *
            "Quiescent protected module $(spatial_patient_simulation.quiescent_protected_module)\n" *
            "Quiescent division rate coeff $(spatial_patient_simulation.quiescent_division_rate_coeff)\n" *
            "Ambient quiescent division rate coeff $(spatial_patient_simulation.ambient_quiescent_division_rate_coeff)\n" *
            "Protected module $(spatial_patient_simulation.protected_module)\n" *
            "Immune response module $(spatial_patient_simulation.immune_response_module)\n" *
            "Immune death rate $(spatial_patient_simulation.immune_death_rate)\n" *
            "Smoking driver module $(spatial_patient_simulation.smoking_driver_module)\n" *
            "Smoking driver fitness augmentation $(spatial_patient_simulation.smoking_driver_fitness_augmentation)"
        )
    end
    return nothing
end

function log_step!(
    spatial_patient_simulation::SpatialPatientSimulation,
    spatial_patient_simulation_record::Union{Records.Record,Nothing},
    current_age::Float64,
    current_year::Ref{UInt8},
    current_record_number::Ref{UInt16},
    csv_logging::Bool,
    record_frequency::Int,
    smoking::Bool,
)::Nothing
    if current_age > current_year[]
        # we've just passed a year boundary
        current_year[] = ceil(UInt8, current_age)
        if current_year[] % 10 == 0
            SpatialCellPopulationClass.recalibrate_current_normalisation_constant!(
                spatial_patient_simulation.cell_population,
                smoking,
            )
            @assert total_mutation_count_matches_sum(spatial_patient_simulation) "total_mutation_count does not match sum of mutation counts"
        end

        if spatial_patient_simulation.allow_early_stopping
            for compartment in values(
                spatial_patient_simulation.cell_population.cell_compartments_dict,
            )
                if compartment.yearly_counters.this_year_immune_death_count /
                   compartment.initial_cell_number > 300
                    if spatial_patient_simulation.logger !== nothing
                        with_logger(spatial_patient_simulation.logger) do
                            @error "Early stopping due to high empirical immune death rate"
                            @debug "Empirical immune death rate $(compartment.yearly_counters.this_year_immune_death_count / compartment.initial_cell_number)"
                        end
                    end
                    throw(
                        InvalidStateException(
                            "TooManyImmuneDeathsError\n" *
                            "Empirical immune death rate $(compartment.yearly_counters.this_year_immune_death_count / compartment.initial_cell_number)\n" *
                            "Current year $(current_year[])\n" *
                            "Current record number $(current_record_number[])\n",
                            :too_many_immune_deaths,
                        ),
                    )
                end
            end
        end

        if csv_logging
            CellPopulationClass.add_yearly_record(
                spatial_patient_simulation.cell_population,
                spatial_patient_simulation_record,
                current_year[],
            )
        elseif spatial_patient_simulation.logger !== nothing
            log_year(spatial_patient_simulation, current_year[])
        end
        if spatial_patient_simulation.cell_population.cell_compartments_dict["main"].yearly_counters !==
           nothing
            CellPopulationClass.reset_yearly_counters!(
                spatial_patient_simulation.cell_population,
            )
        end
    end
    if current_age * record_frequency > current_record_number[]
        # we've just passed a recording boundary
        # note this assumes two haven't been passed in one jump
        current_record_number[] = ceil(UInt16, current_age * record_frequency)
        @assert spatial_patient_simulation.fitness_summaries.row_index[] ==
                (current_record_number[] - 1) * (
            length(spatial_patient_simulation.cell_population.cell_compartments_dict) +
            1
        ) + 1
        @assert spatial_patient_simulation.mutational_burden_record.row_index[] ==
                (current_record_number[] - 1) *
                spatial_patient_simulation.cell_population.cell_lattice.number_of_cells +
                1
        SpatialCellPopulationClass.record_fitness_summary(
            spatial_patient_simulation.cell_population,
            spatial_patient_simulation.fitness_summaries,
            current_record_number[],
        )
        SpatialCellPopulationClass.record_mutational_burden(
            spatial_patient_simulation.cell_population,
            spatial_patient_simulation.mutational_burden_record,
            current_record_number[],
        )
    end
end

function total_mutation_count_matches_sum(
    spatial_patient_simulation::SpatialPatientSimulation,
)
    mut_count_sum = sum(
        spatial_patient_simulation.cell_population.cell_lattice.mutation_counts.total_mutations,
    )
    return spatial_patient_simulation.cell_population.total_mutation_count[] ==
           mut_count_sum
end

function log_year(
    patient_simulation::SpatialPatientSimulation,
    year_number::Integer,
)::Nothing
    with_logger(patient_simulation.logger) do
        @debug (
            "Completed $year_number years\n\t" * join(
                [
                    compartment_name *
                    ": " *
                    SpatialCompartmentClass.yearly_stats_string(compartment) for
                    (compartment_name, compartment) in
                    pairs(patient_simulation.cell_population.cell_compartments_dict)
                ],
                "\n\t",
            )
        )
    end
    return nothing
end

function get_final_mutational_burden(
    spatial_patient_simulation::SpatialPatientSimulation,
)::Tuple{Vector{Int},Vector{Int}}
    return SpatialCellPopulationClass.get_final_mutational_burden(
        spatial_patient_simulation.cell_population,
        false,
    ),
    SpatialCellPopulationClass.get_final_mutational_burden(
        spatial_patient_simulation.cell_population,
        true,
    )
end

end
