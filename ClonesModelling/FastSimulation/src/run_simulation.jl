module RunSimulation

using Random
using Logging
using Statistics
using StatsBase

using ..CellPopulationClass
using ..CondensedPhylogenyClass
using ..PatientSimulationClass
using ..ParameterValuesClass
using ..SmokingRecordClass
using ..TreeBalanceIndex

struct PatientSimulationOutput
    mutational_burden::Vector{Int}
    smoking_signature_mutational_burden::Vector{Int}
    phylogeny_branch_lengths::Vector{Vector{UInt32}}
    simulation_time::Float64
    tree_balance_indices::Union{Vector{Float64},Nothing}
    tree_calculation_time::Union{Float64,Nothing}
    zero_population_error::Bool
    final_cell_count::Int
    min_cell_count::Int
    max_cell_count::Int
end

SimulationOutput = Dict{String,PatientSimulationOutput}

function run_simulations(
    replicate_count::Int,
    include_infants::Bool,
    first_patient_test::Bool,
    status_representative_test::Bool,
    exclude_nature_genetics::Bool,
    supersample_patient_cohort::Bool,
    epidemiology_test_cohort::Bool,
    tree_subsample_count::Int,
    parameter_values::ParameterValuesClass.ParameterValues,
    seed::Int,
    initial_basal_cell_number::Int,
    steps_per_year::Int,
    dynamic_normalisation_power::Float64,
    record_frequency::Int,
    this_probe_logging_directory::Union{String,Nothing},
    console_logging_level::Union{Int,Nothing},
    file_logging_level::Union{Int,Nothing},
    calculate_tree_balance_index::Bool,
    csv_logging::Bool = false,
)::Vector{SimulationOutput}
    smoking_records = SmokingRecordClass.read_smoking_records(
        include_infants,
        first_patient_test,
        status_representative_test,
        exclude_nature_genetics,
        supersample_patient_cohort,
        epidemiology_test_cohort,
    )
    randomiser = Random.Xoshiro(seed)
    initial_protected_population_size::Int = (
        if ParameterValuesClass.is_active(parameter_values, :Protected)
            ceil(
                parameter_values.protected_parameters.protected_fraction *
                initial_basal_cell_number,
            )
        else
            0
        end
    )
    quiescent_protected_population_size::Int = (
        if (
            ParameterValuesClass.is_active(parameter_values, :Quiescent) &&
            ParameterValuesClass.is_active(parameter_values, :Protected)
        )
            ceil(
                parameter_values.quiescent_parameters.quiescent_fraction *
                parameter_values.protected_parameters.protected_fraction *
                initial_basal_cell_number,
            )
        else
            0
        end
    )
    quiescent_main_population_size::Int = (
        if ParameterValuesClass.is_active(parameter_values, :Quiescent)
            ceil(
                parameter_values.quiescent_parameters.quiescent_fraction *
                (initial_basal_cell_number - initial_protected_population_size),
            )
        else
            0
        end
    )
    mutation_driver_probability = ParameterValuesClass.get_mutation_driver_probability(
        parameter_values.fitness_change_scale,
        parameter_values.fitness_change_probability,
        false,
    )

    replicate_simulation_outputs = Vector{SimulationOutput}(undef, replicate_count)
    for replicate_number in 1:replicate_count
        replicate_simulation_outputs[replicate_number] =
            Dict{String,PatientSimulationOutput}()
        for smoking_record in smoking_records
            patient_simulation = PatientSimulationClass.PatientSimulation(
                smoking_record,
                randomiser,
                initial_basal_cell_number - initial_protected_population_size,
                steps_per_year,
                dynamic_normalisation_power,
                record_frequency,
                parameter_values.fitness_change_scale,
                parameter_values.non_smoking_mutations_per_year,
                parameter_values.smoking_mutations_per_year,
                mutation_driver_probability,
                parameter_values.fitness_change_probability,
                ParameterValuesClass.non_smoking_divisions_per_year(parameter_values),
                ParameterValuesClass.smoking_divisions_per_year(parameter_values),
                quiescent_main_population_size,
                if ParameterValuesClass.is_active(parameter_values, :Quiescent)
                    parameter_values.quiescent_parameters.quiescent_divisions_per_year
                else
                    0.0
                end,
                if ParameterValuesClass.is_active(
                    parameter_values,
                    :QuiescentProtected,
                )
                    parameter_values.quiescent_protected_parameters.quiescent_protection_coefficient
                else
                    0.0
                end,
                initial_protected_population_size,
                if ParameterValuesClass.is_active(parameter_values, :Protected)
                    parameter_values.protected_parameters.protection_coefficient
                else
                    0.0
                end,
                quiescent_protected_population_size,
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
                true, # record_phylogenies
                if this_probe_logging_directory !== nothing
                    joinpath(
                        this_probe_logging_directory,
                        "replicate_$(replicate_number-1)",
                    )
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
            simulation_error = false
            simulation_time = @elapsed (
                simulation_error = PatientSimulationClass.run!(
                    patient_simulation,
                    record_frequency,
                    if this_probe_logging_directory !== nothing
                        joinpath(
                            this_probe_logging_directory,
                            "replicate_$(replicate_number-1)",
                        )
                    else
                        nothing
                    end,
                    csv_logging,
                )
            )
            if smoking_record.true_data_cell_count > 10  # otherwise uninteresting tree
                alive_cell_ids = CellPopulationClass.alive_cell_ids(
                    patient_simulation.cell_population,
                )
                cell_population_subsets =
                    PatientSimulationClass.get_cell_population_subsets(
                        alive_cell_ids,
                        uweights(length(alive_cell_ids)),
                        tree_subsample_count,
                        smoking_record.true_data_cell_count,
                        randomiser,
                        false,
                    )
                phylogeny_branch_lengths =
                    Vector{Vector{UInt32}}(undef, length(cell_population_subsets))
                tree_calculation_time = 0.0
                if calculate_tree_balance_index
                    tree_balance_indices =
                        Vector{Float64}(undef, length(cell_population_subsets))
                else
                    tree_balance_indices = nothing
                end
                for (subset_number, cell_population_subset) in
                    enumerate(cell_population_subsets)
                    if length(cell_population_subset) < 10
                        phylogeny_branch_lengths[subset_number] = Vector{UInt32}([])
                        if calculate_tree_balance_index
                            tree_balance_indices[subset_number] = 0.0
                        end
                        continue
                    end
                    tree_calculation_time += @elapsed (
                        condensed_phylogeny =
                            CondensedPhylogenyClass.condense_phylogeny(
                                patient_simulation.cell_population.cell_compartments_dict["main"].cell_phylogeny,
                                patient_simulation.cell_population.cell_compartments_dict["main"].mutation_phylogeny,
                                cell_population_subset,
                            )
                    )
                    phylogeny_branch_lengths[subset_number] =
                        CondensedPhylogenyClass.get_branch_lengths(condensed_phylogeny)
                    if record_frequency > 0
                        open(
                            joinpath(
                                this_probe_logging_directory,
                                "replicate_$(replicate_number-1)",
                                "$(smoking_record.patient).nwk",
                            ),
                            "a",
                        ) do phylogeny_file
                            return println(
                                phylogeny_file,
                                CondensedPhylogenyClass.get_nwk_string(
                                    condensed_phylogeny,
                                    true,
                                ),
                            )
                        end
                    end
                    if calculate_tree_balance_index
                        tree_calculation_time += @elapsed (
                            tree_balance_indices[subset_number] =
                                TreeBalanceIndex.j_one(condensed_phylogeny)
                        )
                    end
                end
            else
                phylogeny_branch_lengths = Vector{Vector{UInt32}}([])
                tree_balance_indices = nothing
                tree_calculation_time = 0.0
            end
            final_mutational_burden, final_smoking_signature_mutational_burden = (
                if simulation_error
                    (Vector{Int}([]), Vector{Int}([]))
                else
                    PatientSimulationClass.get_final_mutational_burden(patient_simulation)
                end
            )
            replicate_simulation_outputs[replicate_number][patient_simulation.smoking_record.patient] =
                PatientSimulationOutput(
                    final_mutational_burden,
                    final_smoking_signature_mutational_burden,
                    phylogeny_branch_lengths,
                    simulation_time,
                    tree_balance_indices,
                    tree_calculation_time,
                    simulation_error,
                    CellPopulationClass.number_of_basal_cells(
                        patient_simulation.cell_population,
                    ),
                    patient_simulation.cell_count_bounds.min,
                    patient_simulation.cell_count_bounds.max,
                )
        end
    end
    return replicate_simulation_outputs
end

function run_simulations(parsed_args::Dict{String,Any})::Vector{SimulationOutput}
    return run_simulations(
        parsed_args["replicate_count"],
        parsed_args["include_infants"],
        parsed_args["first_patient_test"],
        parsed_args["status_representative_test"],
        parsed_args["exclude_nature_genetics"],
        parsed_args["supersample_patient_cohort"],
        parsed_args["epidemiology_test_cohort"],
        parsed_args["tree_subsample_count"],
        ParameterValuesClass.ParameterValues(
            parsed_args["smoking_mutation_rate_augmentation"],
            parsed_args["non_smoking_mutations_per_year"],
            parsed_args["fitness_change_scale"],
            parsed_args["fitness_change_probability"],
            parsed_args["smoking_division_rate_augmentation"],
            parsed_args["non_smoking_divisions_per_year"],
            parsed_args["mutation_rate_multiplier_shape"],
            parsed_args["quiescent_fraction"],
            parsed_args["quiescent_divisions_per_year"],
            parsed_args["ambient_quiescent_divisions_per_year"],
            parsed_args["quiescent_gland_cell_count"],
            parsed_args["quiescent_protection_coefficient"],
            parsed_args["protected_fraction"],
            parsed_args["protection_coefficient"],
            parsed_args["protected_region_radius"],
            parsed_args["immune_death_rate"],
            parsed_args["smoking_immune_coeff"],
            parsed_args["smoking_driver_fitness_augmentation"],
        ),
        parsed_args["seed"],
        parsed_args["initial_basal_cell_number"],
        parsed_args["steps_per_year"],
        parsed_args["dynamic_normalisation_power"],
        parsed_args["record_frequency"],
        parsed_args["this_probe_logging_directory"],
        parsed_args["console_logging_level"],
        parsed_args["file_logging_level"],
        parsed_args["calculate_tree_balance_index"],
        parsed_args["csv_logging"],
    )
end

function test_run(
    initial_basal_cell_number::Int = 400,
    seed::Int = 2021,
    make_logs::Bool = true,
)
    parameter_values = ParameterValuesClass.ParameterValues(
        100, # smoking_mutations_per_year
        25, # non_smoking_mutations_per_year
        0.1, # fitness_change_scale
        0.0174, # fitness_change_probability  
        59.4, # smoking_divisions_per_year
        33.0, # non_smoking_divisions_per_year 
        nothing, # mutation_rate_multiplier_shape
        # ParameterValuesClass.QuiescentParameters(0.05, 4),
        nothing, # quiescent_parameters
        # ParameterValuesClass.QuiescentProtectedParameters(0.5),
        nothing, # quiescent_protected_parameters
        nothing, # protected_parameters
        ParameterValuesClass.ImmuneResponseParameters(0.0001, 0.01),
        ParameterValuesClass.SmokingDriverParameters(1),
    )
    steps_per_year = 365
    dynamic_normalisation_power = 10.0
    record_frequency = 10
    csv_logging = !make_logs
    return run_simulation(
        false,
        false,
        true,
        true,
        parameter_values,
        seed,
        initial_basal_cell_number,
        steps_per_year,
        dynamic_normalisation_power,
        record_frequency,
        "test_run",
        if make_logs
            0
        else
            nothing
        end,
        if make_logs
            -1000
        else
            nothing
        end,
        csv_logging,
    )
end

function timings_test_run(
    grid_side_lengths::Union{Vector{Int},UnitRange{Int}},
    seeds::Union{Vector{Int},UnitRange{Int}},
    output_dir::String = "non_spatial_simulation_test_run",
    silent::Bool = false,
)::Nothing
    mkpath(output_dir * "/runs")
    if !isfile("$output_dir/timings.txt")
        open("$output_dir/timings.txt", "w") do f
            return println(f, "grid_side_length,seed,time,std")
        end
    end
    timings = Vector{Float64}(undef, length(grid_side_lengths))
    Threads.@threads for ((i, grid_side_length), seed) in
                         collect(Iterators.product(enumerate(grid_side_lengths), seeds))
        println(
            "Running test run with 'grid side length' $(grid_side_length) and seed $(seed)",
        )
        timings[i] = @elapsed (output = test_run(grid_side_length^2, seed, !silent))
        std = Statistics.std(output["PD26988"].mutational_burden)
        println("Standard deviation: $(std), time: $(timings[i])")
        open("$output_dir/timings.txt", "a") do f
            return println(f, "$(grid_side_length), $(seed), $(timings[i]), $(std)")
        end
        # print mb to a file
        open("$output_dir/runs/mb_$(grid_side_length)_$(seed).txt", "w") do f
            for mb in output["PD26988"].mutational_burden
                println(f, mb)
            end
        end
    end
    return nothing
end

end
