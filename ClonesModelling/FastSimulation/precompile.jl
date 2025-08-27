using FastSimulation

parsed_args = FastSimulation.ParseCmdLineArgs.parse_cmd_line_args([
    "--record_phylogenies",
    "--initial_basal_cell_number",
    "100",
    "--exclude_nature_genetics",
    "--replicate_count",
    "1",
])
FastSimulation.RunSimulation.run_simulations(parsed_args)
parsed_args["spatial"] = true
FastSimulation.RunSpatialSimulation.run_spatial_simulations(parsed_args)
