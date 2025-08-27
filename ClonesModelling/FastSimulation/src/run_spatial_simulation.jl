module RunSpatialSimulation

using Logging
using Random
using Statistics
using Base.Threads

using ..CellLatticeClass
using ..CellPhylogenyClass
using ..CondensedPhylogenyClass
using ..MutationPhylogenyClass
using ..ParameterValuesClass
using ..PatientSimulationClass
using ..SmokingRecordClass
using ..SpatialPatientSimulationClass
using ..TreeBalanceIndex

struct SpatialPatientSimulationOutput
    mutational_burden::Vector{Int}
    smoking_signature_mutational_burden::Union{Vector{Int},Nothing}
    phylogeny_branch_lengths::Union{Vector{Vector{UInt32}},Nothing}
    simulation_time::Float64
    tree_balance_indices::Union{Vector{Float64},Nothing}
    tree_calculation_time::Union{Float64,Nothing}
end

SpatialSimulationOutput = Dict{String,SpatialPatientSimulationOutput}

function run_spatial_simulations(
    replicate_count::Int,
    include_infants::Bool,
    first_patient_test::Bool,
    status_representative_test::Bool,
    exclude_nature_genetics::Bool,
    supersample_patient_cohort::Bool,
    epidemiology_test_cohort::Bool,
    tree_subsample_count::Union{Int,Nothing},
    parameter_values::ParameterValuesClass.ParameterValues,
    seed::Int,
    grid_side_length::Int,
    record_frequency::Int,
    this_probe_logging_directory::Union{String,Nothing},
    console_logging_level::Union{Int,Nothing},
    file_logging_level::Union{Int,Nothing},
    calculate_tree_balance_index::Bool,
    csv_logging::Bool = false,
    allow_early_stopping::Bool = false,
    record_phylogenies::Bool = false,
)::Vector{SpatialSimulationOutput}
    smoking_records = SmokingRecordClass.read_smoking_records(
        include_infants,
        first_patient_test,
        status_representative_test,
        exclude_nature_genetics,
        supersample_patient_cohort,
        epidemiology_test_cohort,
    )
    randomiser = Random.Xoshiro(seed)

    quiescent_spacing, protected_spacing =
        get_lattice_spacing(parameter_values, grid_side_length)

    record_phylogenies = calculate_tree_balance_index || record_phylogenies

    replicate_simulation_outputs =
        Vector{SpatialSimulationOutput}(undef, replicate_count)

    for replicate_number in 1:replicate_count
        replicate_simulation_outputs[replicate_number] =
            Dict{String,SpatialPatientSimulationOutput}()
        for smoking_record in smoking_records
            spatial_patient_simulation =
                SpatialPatientSimulationClass.SpatialPatientSimulation(
                    smoking_record,
                    randomiser,
                    grid_side_length,
                    record_frequency,
                    quiescent_spacing,
                    protected_spacing,
                    parameter_values,
                    csv_logging,
                    allow_early_stopping,
                    record_phylogenies,
                    this_probe_logging_directory,
                    replicate_number,
                    console_logging_level,
                    file_logging_level,
                )
            simulation_error = false
            simulation_time = @elapsed (SpatialPatientSimulationClass.run!(
                spatial_patient_simulation,
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
            ))
            if record_phylogenies && smoking_record.true_data_cell_count > 10
                alive_cell_ids, alive_cell_weights =
                    (CellLatticeClass.get_alive_cell_ids(
                        spatial_patient_simulation.cell_population.cell_lattice,
                        spatial_patient_simulation.cell_population.quiescent_gland_cell_count,
                    ))
                cell_population_subsets =
                    PatientSimulationClass.get_cell_population_subsets(
                        alive_cell_ids,
                        alive_cell_weights,
                        tree_subsample_count,
                        smoking_record.true_data_cell_count,
                        randomiser,
                        true,
                    )
                phylogeny_branch_lengths =
                    Vector{Vector{UInt32}}(undef, length(cell_population_subsets))
                tree_calculation_time = 0.0
                tree_balance_indices = (
                    if calculate_tree_balance_index
                        Vector{Float64}(undef, length(cell_population_subsets))
                    else
                        nothing
                    end
                )
                for (subset_number, cell_population_subset) in
                    enumerate(cell_population_subsets)
                    tree_calculation_time += @elapsed (
                        condensed_phylogeny =
                            CondensedPhylogenyClass.condense_phylogeny(
                                spatial_patient_simulation.cell_population.cell_compartments_dict["main"].cell_phylogeny,
                                spatial_patient_simulation.cell_population.cell_compartments_dict["main"].mutation_phylogeny,
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
                phylogeny_branch_lengths = nothing
                tree_calculation_time = nothing
                tree_balance_indices = nothing
            end
            final_mutational_burden, final_smoking_signature_mutational_burden = (
                if simulation_error
                    (Vector{Int}([]), Vector{Int}([]))
                else
                    SpatialPatientSimulationClass.get_final_mutational_burden(
                        spatial_patient_simulation,
                    )
                end
            )
            replicate_simulation_outputs[replicate_number][smoking_record.patient] =
                SpatialPatientSimulationOutput(
                    final_mutational_burden,
                    final_smoking_signature_mutational_burden,
                    phylogeny_branch_lengths,
                    simulation_time,
                    tree_balance_indices,
                    tree_calculation_time,
                )
        end
    end
    return replicate_simulation_outputs
end

function run_spatial_simulations(
    parsed_args::Dict{String,Any},
)::Vector{SpatialSimulationOutput}
    return run_spatial_simulations(
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
        ceil(Int, sqrt(parsed_args["initial_basal_cell_number"])),
        parsed_args["record_frequency"],
        parsed_args["this_probe_logging_directory"],
        parsed_args["console_logging_level"],
        parsed_args["file_logging_level"],
        parsed_args["calculate_tree_balance_index"],
        parsed_args["csv_logging"],
        parsed_args["allow_early_stopping"],
        parsed_args["record_phylogenies"],
    )
end

function get_lattice_spacing(
    parameter_values::ParameterValuesClass.ParameterValues,
    grid_side_length::Int,
)::Tuple{Int,Int}
    quiescent_spacing::Int = (
        if ParameterValuesClass.is_active(parameter_values, :Quiescent)
            ceil(
                1 / sqrt(
                    parameter_values.quiescent_parameters.quiescent_fraction /
                    parameter_values.quiescent_parameters.quiescent_gland_cell_count,
                ),
            )
        else
            0
        end
    )
    if quiescent_spacing > grid_side_length
        # println(
        #     "Warning: quiescent spacing is greater than grid side length; setting to 0.8*grid_side_length",
        # )
        quiescent_spacing = ceil(0.8 * grid_side_length)
    end

    protected_spacing::Int = (
        if ParameterValuesClass.is_active(parameter_values, :Protected)
            get_protected_spacing(
                parameter_values.protected_parameters.protected_region_radius,
                parameter_values.protected_parameters.protected_fraction,
            )
        else
            0
        end
    )
    if protected_spacing > grid_side_length
        # print(
        #     "Warning: protected spacing $protected_spacing is greater than grid side " *
        #     "length $grid_side_length; setting to 0.8*grid_side_length = ",
        # )
        protected_spacing = ceil(0.8 * grid_side_length)
        # println(protected_spacing)
    end
    return (quiescent_spacing, protected_spacing)
end

function test_run(
    grid_side_length::Int = 10,
    seed::Int = 1,
    filename::Union{String,Nothing} = nothing,
    protected_fraction::Float64 = 0.1,
    protected_region_radius::Int = 2,
    quiescent_fraction::Float64 = 0.1,
    silent::Bool = false,
    calculate_tree_balance_index::Bool = true,
    make_records::Bool = true,
    status_representative_test::Bool = true,
    record_phylogenies::Bool = false,
    quiescent_protection_coefficient::Float64 = 1.0,
    csv_logging::Bool = false,
)
    include_infants = true
    first_patient_test = false
    exclude_nature_genetics = true
    parameter_values = ParameterValuesClass.ParameterValues(
        3.0, # smoking_mutation_rate_augmentation
        25.0, # non_smoking_mutations_per_year
        0.1, # fitness_change_scale
        0.0174, # fitness_change_probability
        2.0, # smoking_division_rate_augmentation
        5.0, # non_smoking_divisions_per_year
        nothing, # mutation_rate_multiplier_shape
        quiescent_fraction, # quiescent_fraction
        3.0, # quiescent_divisions_per_year
        3.0, # ambient_quiescent_divisions_per_year
        quiescent_protection_coefficient, # quiescent_protection_coefficient
        protected_fraction, # protected_fraction
        0.01, # protection_coefficient
        protected_region_radius, # protected_region_radius
        0.0001, # immune_death_rate
        0.01, # smoking_immune_coeff
        70.0, # smoking_driver_fitness_augmentation
    )
    record_frequency = make_records ? 3 : 0
    this_probe_logging_directory = if silent
        nothing
    elseif filename === nothing
        "spatial_test_run"
    else
        filename
    end
    console_logging_level = (
        if silent
            nothing
        else
            -1000 # debug
        end
    )
    file_logging_level = (
        if silent
            nothing
        else
            -1000 # debug
        end
    )
    allow_early_stopping = false
    sim_output = run_spatial_simulations(
        1,
        include_infants,
        first_patient_test,
        status_representative_test,
        exclude_nature_genetics,
        false, # supersample_patient_cohort
        nothing, # tree_subsample_count
        parameter_values,
        seed,
        grid_side_length,
        record_frequency,
        this_probe_logging_directory,
        console_logging_level,
        file_logging_level,
        calculate_tree_balance_index,
        csv_logging,
        allow_early_stopping,
        record_phylogenies,
    )
    return sim_output
end

function timings_test_run(
    grid_side_lengths::Union{Vector{Int},UnitRange{Int}},
    seeds::Union{Vector{Int},UnitRange{Int}},
    remove_modules::Bool = false,
    output_dir::String = "spatial_simulation_test_run",
    silent::Bool = false,
    threads::Bool = false,
)::Nothing
    mkpath(output_dir * "/runs")
    protected_fraction, quiescent_fraction = (
        if remove_modules
            (0.0, 0.0)
        else
            (0.1, 0.1)
        end
    )
    # if timings file doesn't exist, make it and add columns 
    if !isfile(output_dir * "/timings.txt")
        open(output_dir * "/timings.txt", "w") do timings_csv_file
            return println(timings_csv_file, "grid_side_length,seed,timing,std")
        end
    end
    timings = Vector{Float64}(undef, length(grid_side_lengths))

    function timings_test_run_loop(i::Int, grid_side_length::Int, seed::Int)::Nothing
        println(
            "Running spatial test run with grid side length $(grid_side_length) and seed $(seed)",
        )
        timings[i] = @elapsed sim_output = test_run(
            grid_side_length,
            seed,
            "$output_dir/runs/mb_$(grid_side_length)_$(seed).txt",
            protected_fraction,
            2,
            quiescent_fraction,
            silent,
        )
        std = Statistics.std(sim_output["PD26988"].mutational_burden)
        open(output_dir * "/timings.txt", "a") do timings_csv_file
            return println(
                timings_csv_file,
                "$(grid_side_length), $(seed), $(timings[i]), $(std)",
            )
        end
    end
    if !threads
        for (i, grid_side_length) in enumerate(grid_side_lengths)
            for seed in seeds
                timings_test_run_loop(i, grid_side_length, seed)
            end
        end
    else
        Threads.@threads for ((i, grid_side_length), seed) in collect(
            Iterators.product(enumerate(grid_side_lengths), seeds),
        )
            timings_test_run_loop(i, grid_side_length, seed)
        end
    end
    return nothing
end

function get_protected_spacing(
    protected_region_radius::Int,
    protected_fraction::Float64,
)::Int
    protected_region_area =
        2 * protected_region_radius^2 + 2 * protected_region_radius + 1
    return round(sqrt(protected_region_area / protected_fraction))
end
end
