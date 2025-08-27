module ParseCmdLineArgs

using ArgParse

function parse_cmd_line_args(args::Vector{String} = ARGS)::Dict{String,Any}
    arg_parse_settings = ArgParseSettings()
    @add_arg_table arg_parse_settings begin
        "--spatial"
        help = "Run simulation on 2d lattice"
        action = :store_true
    end

    # args in common between spatial and non-spatial
    @add_arg_table arg_parse_settings begin
        "--replicate_count"
        help = "Number of simulations to run per patient, at the same parameter values"
        default = 1
        arg_type = Int

        "--include_infants"
        help = "Simulate patients under 5 years old"
        action = :store_true

        "--first_patient_test"
        help = "Simulate only the first patient, for testing"
        action = :store_true

        "--status_representative_test"
        help = "Simulate a representative of each smoking history type"
        action = :store_true

        "--exclude_nature_genetics"
        help = "Exclude patients from Nature Genetics paper"
        action = :store_true

        "--supersample_patient_cohort"
        help = "Use larger synthetic patient cohort"
        action = :store_true

        "--epidemiology_test_cohort"
        help = "Use synthetic grid-search patient cohort for epidemiology test"
        action = :store_true

        "--seed"
        help = "Random seed"
        default = 1
        arg_type = Int

        "--initial_basal_cell_number"
        help = "Initial number of basal cells"
        default = 100
        arg_type = Int

        "--record_frequency"
        help = "Number of records made per simulated year"
        default = 0
        arg_type = Int

        "--this_probe_logging_directory"
        help = "Directory to log this probe's output"
        default = nothing
        arg_type = String

        "--console_logging_level"
        help = "Logging level for console output, as in Logging.jl"
        default = nothing
        arg_type = Int

        "--file_logging_level"
        help = "Logging level for file output, as in Logging.jl"
        default = nothing
        arg_type = Int

        "--calculate_tree_balance_index"
        help = "Calculate tree balance index for each simulated tree"
        action = :store_true

        "--tree_subsample_count"
        help = "Number of trees to subsample for tree balance index calculation"
        default = 100
        arg_type = Int

        "--csv_logging"
        help = "Log yearly cell counts to csv files rather than text, to save space"
        action = :store_true

        "--allow_early_stopping"
        help = "Allow simulation to stop early for biological implausbility"
        action = :store_true

        "--record_phylogenies"
        help = "Record phylogenies"
        action = :store_true
    end

    # args specific to non-spatial simulation
    @add_arg_table arg_parse_settings begin
        "--steps_per_year"
        help = "Number of simulation steps per year in non-spatial simulation"
        default = 100
        arg_type = Int

        "--dynamic_normalisation_power"
        help = "Parameter affecting the degree to which population is forced to be constant"
        default = 10.0
        arg_type = Float64
    end

    # parameter values
    @add_arg_table arg_parse_settings begin
        "--smoking_mutation_rate_augmentation"
        help = "Proportional increase in mutation rate while smoking"
        default = 3.0
        arg_type = Float64

        "--non_smoking_mutations_per_year"
        help = "Number of mutations per year while not smoking"
        default = 25.0
        arg_type = Float64

        "--fitness_change_scale"
        help = "Coefficient of fitness change distribution"
        default = 0.05
        arg_type = Float64

        "--fitness_change_probability"
        help = "Probability of non-zero fitness change in fitness change distribution"
        default = 0.1
        arg_type = Float64

        "--smoking_division_rate_augmentation"
        help = "Proportional increase in division rate while smoking"
        default = 0.8
        arg_type = Float64

        "--non_smoking_divisions_per_year"
        help = "Number of divisions per year while not smoking"
        default = 1.6
        arg_type = Float64

        "--mutation_rate_multiplier_shape"
        help = "Shape parameter of gamma distribution for mutation rate multiplier"
        default = nothing
        arg_type = Union{Float64,Nothing}

        "--quiescent_fraction"
        help = "number of initial quiescent cells, as a fraction of the initial basal cell number"
        default = 0.1
        arg_type = Float64

        "--quiescent_divisions_per_year"
        help = "Number of divisions per year while quiescent"
        default = 1.6
        arg_type = Float64

        "--ambient_quiescent_divisions_per_year"
        help = "non-productive quiescent cell divisions per year"
        default = 1.0
        arg_type = Float64

        "--quiescent_gland_cell_count"
        help = "number of cells in each quiescent gland"
        default = 10
        arg_type = Int

        "--quiescent_protection_coefficient"
        help = "degree of protection for quiescent cells"
        default = 0.5
        arg_type = Float64

        "--protected_fraction"
        help = "fraction of cells protected from smoking"
        default = 0.1
        arg_type = Float64

        "--protection_coefficient"
        help = "degree of protection from smoking"
        default = 0.5
        arg_type = Float64

        "--protected_region_radius"
        help = "radius of each protected region in spatial simulation"
        default = 2
        arg_type = Int

        "--immune_death_rate"
        help = "number of immune deaths per mutation per year"
        default = 0.0001
        arg_type = Float64

        "--smoking_immune_coeff"
        help = "effect of smoking on immune death"
        default = 0.01
        arg_type = Float64

        "--smoking_driver_fitness_augmentation"
        help = "fitness augmentation for driver mutations while smoking"
        default = 1.0
        arg_type = Float64
    end

    return parse_args(args, arg_parse_settings)
end

end
