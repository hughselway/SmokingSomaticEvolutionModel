using FastSimulation

parsed_args = FastSimulation.ParseCmdLineArgs.parse_cmd_line_args()
replicate_simulation_outputs = (
    if parsed_args["spatial"]
        FastSimulation.RunSpatialSimulation.run_spatial_simulations(parsed_args)
    else
        FastSimulation.RunSimulation.run_simulations(parsed_args)
    end
)

# print output to stdout for the process caller to capture
for (replicate_number, simulation_output) in enumerate(replicate_simulation_outputs)
    for (patient, patient_simulation_output) in simulation_output
        println(
            join(
                vcat(
                    [
                        replicate_number,
                        patient,
                        join(patient_simulation_output.mutational_burden, " "),
                        join(
                            patient_simulation_output.smoking_signature_mutational_burden,
                            " ",
                        ),
                        if (
                            patient_simulation_output.phylogeny_branch_lengths ===
                            nothing
                        )
                            ""
                        else
                            join(
                                [
                                    join(Int.(subset_branch_lengths), " ") for
                                    subset_branch_lengths in
                                    patient_simulation_output.phylogeny_branch_lengths
                                ],
                                ";",
                            )
                        end,
                        patient_simulation_output.simulation_time,
                        if patient_simulation_output.tree_balance_indices === nothing
                            ""
                        else
                            join(patient_simulation_output.tree_balance_indices, " ")
                        end,
                        if patient_simulation_output.tree_calculation_time === nothing
                            ""
                        else
                            patient_simulation_output.tree_calculation_time
                        end,
                    ],
                    if parsed_args["spatial"]
                        []
                    else
                        [
                            patient_simulation_output.zero_population_error,
                            patient_simulation_output.final_cell_count,
                            patient_simulation_output.min_cell_count,
                            patient_simulation_output.max_cell_count,
                        ]
                    end,
                ),
                "--",
            ),
        )
    end
end
