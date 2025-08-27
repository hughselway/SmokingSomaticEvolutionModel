using Random
using Statistics
using DataFrames
using Distributions
using Roots
using Plots
using Plots.PlotMeasures
using CSV
using JSON

# Simulate birth-death process for a fixed time M
function simulate_birth_death_process_fixed_time(
    b::Float64,
    d::Float64,
    M::Int,
    num_simulations::Int,
    spatial::Bool,
    cell_count_cap::Int,
)::Tuple{Vector{Int},Vector{Bool}}
    final_states = zeros(num_simulations)  # Store the final state at time M
    absorbed_at_0 = falses(num_simulations)  # Store whether the chain was absorbed at 0

    Threads.@threads for i in 1:num_simulations
        n = 1  # Start at state n = 1
        time = 0.0

        while time < M && n > 0
            if n > cell_count_cap
                final_states[i] = n
                break
            end

            birth_rate = spatial ? b * sqrt(n) : b * n
            death_rate = spatial ? d * sqrt(n) : d * n
            total_rate = birth_rate + death_rate

            # Time until the next event (exponential distribution)
            time_to_next_event = rand(Exponential(1 / total_rate))

            if time + time_to_next_event >= M
                # If the next event exceeds the time limit M, stop and record the state
                final_states[i] = n
                break
            end

            # Otherwise, update time and state
            time += time_to_next_event

            # Randomly decide to go up (birth) or down (death)
            if rand() < birth_rate / total_rate
                n += 1  # Move up
            else
                n -= 1  # Move down
            end
        end

        # If we ended up at state 0, mark it as absorbed
        if n == 0
            absorbed_at_0[i] = true
        else
            final_states[i] = n
        end
    end

    return final_states, absorbed_at_0
end

# Perform analysis to compute the average state at time M (for non-absorbed trajectories) and absorption probability
function analyze_simulations(
    final_states::Vector{Int},
    absorbed_at_0::Vector{Bool},
)::Tuple{Float64,Float64,Float64,Float64,Float64}
    # Probability of absorption at state 0
    prob_absorption = mean(absorbed_at_0)

    # Average state at time M, excluding absorbed states
    non_absorbed_states = final_states[.!absorbed_at_0]
    # non_absorbed_states = final_states
    if length(non_absorbed_states) > 0
        average_state_at_M = mean(non_absorbed_states)
        median_state_at_M = median(non_absorbed_states)
        # get 5th and 95th percentiles  
        sorted_states = sort(non_absorbed_states)
        fifth_percentile = sorted_states[round(Int, 0.05 * length(sorted_states))]
        ninetyfifth_percentile = sorted_states[round(Int, 0.95 * length(sorted_states))]
    else
        average_state_at_M = NaN  # In case all trajectories were absorbed
        median_state_at_M = NaN
        fifth_percentile = NaN
        ninetyfifth_percentile = NaN
    end

    return prob_absorption,
    average_state_at_M,
    median_state_at_M,
    fifth_percentile,
    ninetyfifth_percentile
end

function get_birth_death_rates(F::Float64, spatial::Bool)::Tuple{Float64,Float64}
    print("F=", round(F, digits = 10), " ")
    if spatial
        A = sqrt(pi)
        birth_rate = A * (1 + F) / (4 + F)
        death_rate = A * (1 - F) / (4 + 3F)
    else
        birth_rate = (1 + F) / 2
        death_rate = (1 - F) / 2
    end
    return birth_rate, death_rate
end

## Parameters
M = 60  # Fixed time to run the simulation; meant to be a reasonable/mean/median age for somatic mutations
num_simulations = 10^5  # Number of simulations
F_values = 0:0.01:1.0  # Projected fitness values F
cell_count_cap = 10^4

# we say a clone is significantly selected for if it reaches above the 95th percentile
# of the clone size distribution at F=0. We then find the F value at which the mean 
# clone size reaches the threshold 

prob_absorption_values = Dict{Bool,Vector{Float64}}()
average_state_at_M_values = Dict{Bool,Vector{Float64}}()
median_state_at_M_values = Dict{Bool,Vector{Float64}}()
fifth_percentile_values = Dict{Bool,Vector{Float64}}()
ninetyfifth_percentile_values = Dict{Bool,Vector{Float64}}()

n_cells_threshold = Dict{Bool,Integer}()
F_threshold = Dict{Bool,Float64}()
absorption_prob_at_threshold = Dict{Bool,Float64}()

results_dir = "notebooks/mdp_model_bootstrap/results/$(M)_years"
if !isdir(results_dir)
    mk(results_dir)
end
for spatial in [true, false]
    bootstrap_results_file = "$results_dir/bootstrap_results_sp=$spatial.csv"
    threshold_json_file = "$results_dir/F_threshold_sp=$spatial.json"
    if !isfile(bootstrap_results_file)
        prob_absorption_values[spatial] = zeros(length(F_values))
        average_state_at_M_values[spatial] = zeros(length(F_values))
        median_state_at_M_values[spatial] = zeros(length(F_values))
        fifth_percentile_values[spatial] = zeros(length(F_values))
        ninetyfifth_percentile_values[spatial] = zeros(length(F_values))
        ## Run plotting simulations
        for (i, F) in enumerate(F_values)
            B, C = get_birth_death_rates(F, spatial)

            # Run the simulation
            final_states, absorbed_at_0 = simulate_birth_death_process_fixed_time(
                B,
                C,
                M,
                num_simulations,
                spatial,
                cell_count_cap,
            )

            # Perform the analysis
            (
                prob_absorption,
                average_state_at_M,
                median_state_at_M,
                fifth_percentile,
                ninetyfifth_percentile,
            ) = analyze_simulations(final_states, absorbed_at_0)
            prob_absorption_values[spatial][i] = prob_absorption
            average_state_at_M_values[spatial][i] = average_state_at_M
            median_state_at_M_values[spatial][i] = median_state_at_M
            fifth_percentile_values[spatial][i] = fifth_percentile
            ninetyfifth_percentile_values[spatial][i] = ninetyfifth_percentile
        end
        println()
        open(bootstrap_results_file, "w") do f
            write(
                f,
                "F,prob_absorption,average_state_at_M,median_state_at_M,fifth_percentile,ninetyfifth_percentile\n",
            )
            for i in eachindex(F_values)
                write(
                    f,
                    "$(F_values[i]),$(prob_absorption_values[spatial][i]),$(average_state_at_M_values[spatial][i]),$(median_state_at_M_values[spatial][i]),$(fifth_percentile_values[spatial][i]),$(ninetyfifth_percentile_values[spatial][i])\n",
                )
            end
        end
    else
        results = CSV.read(bootstrap_results_file, DataFrame)
        @assert F_values == results.F
        prob_absorption_values[spatial] = results.prob_absorption
        average_state_at_M_values[spatial] = results.average_state_at_M
        median_state_at_M_values[spatial] = results.median_state_at_M
        fifth_percentile_values[spatial] = results.fifth_percentile
        ninetyfifth_percentile_values[spatial] = results.ninetyfifth_percentile
    end

    if !isfile(threshold_json_file)
        n_cells_threshold[spatial] = ninetyfifth_percentile_values[spatial][1]
        print("n_cells_threshold: $(n_cells_threshold[spatial])")
        F_threshold[spatial] = find_zero(
            F -> begin
                final_states = simulate_birth_death_process_fixed_time(
                    get_birth_death_rates(F, spatial)...,
                    M,
                    num_simulations,
                    spatial,
                    cell_count_cap,
                )[1]
                non_zero_states = final_states[final_states.>0]
                mean(non_zero_states) - n_cells_threshold[spatial]
                # mean(final_states) - n_cells_threshold[spatial]
            end,
            0.1,
        )
        absorption_prob_at_threshold[spatial] =
            simulate_birth_death_process_fixed_time(
                get_birth_death_rates(F_threshold[spatial], spatial)...,
                M,
                num_simulations,
                spatial,
                cell_count_cap,
            )[2] |> mean
        open(threshold_json_file, "w") do f
            return write(
                f,
                "{\"F_threshold\": $(F_threshold[spatial])," *
                "\"absorption_prob_at_threshold\": $(absorption_prob_at_threshold[spatial])," *
                "\"n_cells_threshold\": $(n_cells_threshold[spatial])}\n",
            )
        end
    else
        threshold_json = JSON.parsefile(threshold_json_file)
        F_threshold[spatial] = threshold_json["F_threshold"]
        absorption_prob_at_threshold[spatial] =
            threshold_json["absorption_prob_at_threshold"]
        n_cells_threshold[spatial] = threshold_json["n_cells_threshold"]
    end
    println(
        "\nspatial = $spatial\n" *
        "\tn_cells_threshold = $(n_cells_threshold[spatial])\n" *
        "\tF_threshold = $(F_threshold[spatial])\n" *
        "\tabsorption_prob_at_threshold = $(absorption_prob_at_threshold[spatial])",
    )
end

# Plot results
plot_dir = "notebooks/mdp_model_bootstrap/plots/$(M)_years"
if !isdir(plot_dir)
    mkdir(plot_dir)
end

for (spatial, log_scale_y) in [(true, false), (false, false), (false, true)]
    plot(
        F_values,
        average_state_at_M_values[spatial],
        label = "Mean",
        xlabel = "F (sigmoid-projected fitness)",
        ylabel = "Cell count" * (log_scale_y ? " (log scale)" : ""),
        title = "Clone size after $(M) years (" * (spatial ? "" : "non-") * "spatial)",
        legend = spatial ? :topleft : :right,
        lw = 2,
        size = (500, 300),
        fmt = :pdf,
        left_margin = 2mm,
        bottom_margin = 2mm,
    )
    plot!(F_values, median_state_at_M_values[spatial], label = "Median", lw = 2)
    #dotted lines for 5th and 95th percentiles
    plot!(
        F_values,
        fifth_percentile_values[spatial],
        label = "5th, 95th percentiles",
        linestyle = :dot,
        lw = 2,
        colour = :black,
    )
    plot!(
        F_values,
        ninetyfifth_percentile_values[spatial],
        label = "",
        linestyle = :dot,
        lw = 2,
        colour = :black,
    )
    if !spatial
        plot!(
            F_values,
            min.(
                ((1 .+ F_values) .* exp.(M .* F_values) .- (1 .- F_values)) ./
                (2 .* F_values),
                cell_count_cap,
            ),
            label = "Analytical mean",
            linestyle = :dot,
            lw = 2,
            colour = :red,
        )
    end

    if log_scale_y
        yaxis!(:log)
    end
    hline!([n_cells_threshold[spatial]], label = "", lw = 2, colour = :grey)
    vline!([F_threshold[spatial]], label = "", lw = 2, colour = :grey)
    ymax = maximum(ninetyfifth_percentile_values[spatial])
    annotate!(
        spatial ? 0.0 : 1.0,
        ymax * (log_scale_y ? 0.08 : 0.7),
        text(
            "$(n_cells_threshold[spatial]) cells passed at " *
            "F=$(round(F_threshold[spatial],digits=2))",
            spatial ? :left : :right,
            9,
        ),
    )
    if !log_scale_y
        savefig("$plot_dir/average_state_at_M_sp=$spatial.pdf")
    else
        savefig("$plot_dir/average_state_at_M_sp=$(spatial)_log_scale.pdf")
    end

    plot(
        F_values,
        prob_absorption_values[spatial],
        # label = "Probability of clone death (absorption at 0) after $(M) years",
        label = "",
        xlabel = "F (sigmoid-projected fitness)",
        ylabel = "Frequency of clone death",
        title = "Clone death by $M years (" * (spatial ? "" : "non-") * "spatial)",
        legend = :topright,
        lw = 2,
        size = (500, 300),
        fmt = :pdf,
        left_margin = 2mm,
        bottom_margin = 2mm,
    )
    vline!([F_threshold[spatial]], label = "", lw = 2, colour = :grey)
    hline!([absorption_prob_at_threshold[spatial]], label = "", lw = 2, colour = :grey)
    annotate!(
        0.0,
        0.1,
        text(
            "p=$(round(100*absorption_prob_at_threshold[spatial],digits=1))% at " *
            "F=$(round(F_threshold[spatial],digits=2))",
            :left,
            10,
        ),
    )
    savefig("$plot_dir/prob_absorption_sp=$spatial.pdf")
end
