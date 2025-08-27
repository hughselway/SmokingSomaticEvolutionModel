module FastSimulation
# without dependencies:
include("./current_fitness_class.jl")
include("./mutation_count_class.jl")
include("./records.jl")
include("./smoking_affected_parameter_class.jl")
include("./smoking_record_class.jl")
include("./parameter_values_class.jl")
include("./parse_cmd_line_args.jl")

# with dependencies:
include("./mutation_phylogeny_class.jl")
include("./cell_phylogeny_class.jl")
include("./compartment_class.jl")
include("./cell_population_class.jl")

# spatial:
include("./cell_lattice_class.jl")
include("./spatial_cell_phylogeny_class.jl")
include("./condensed_phylogeny_class.jl")

# run_simulation requires condensed_phylogeny:
include("./patient_simulation_class.jl")
include("./tree_balance_index.jl")
include("./run_simulation.jl")

# remaining spatial:
include("./spatial_compartment_class.jl")
include("./spatial_cell_population_class.jl")
include("./spatial_patient_simulation_class.jl")
include("./run_spatial_simulation.jl")
end
