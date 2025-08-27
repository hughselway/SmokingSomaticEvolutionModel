module MutationPhylogenyClass

import Base: getindex, string

using ..MutationCountClass

struct MutationalProfile
    parent_profile_id::Union{UInt32,Nothing}
    mutation_fitness::Float64
    smoking_signature::Bool
    step_of_origin::UInt16
    cell_of_origin::UInt64
end

function string(mp::MutationalProfile)::String
    return (
        "p$(mp.parent_profile_id),f$(mp.mutation_fitness)" *
        (mp.smoking_signature ? ",ss" : ",nss") *
        ",o$(mp.step_of_origin),c$(mp.cell_of_origin)"
    )
end

mutable struct MutationPhylogeny
    # indexed by index[], equivalent to profile_id - profiles_printed_to_disk[]
    mutational_profile_list::Vector{MutationalProfile}
    # indexed by index[] + profiles_printed_to_disk[], equivalent to profile_id
    mutation_count_list::Vector{MutationCountClass.MutationCount}
    # TODO remove mutation count list for spatial simulations
    index::UInt64
    max_index::UInt64
    profiles_printed_to_disk::UInt64
end

function string(mp::MutationPhylogeny)::String
    return "MutationPhylogeny:\n\t" * join(
        [
            "$i: " *
            string(mp.mutational_profile_list[i]) *
            ",$(mp.mutation_count_list[i].driver_smoking_signature)," *
            "$(mp.mutation_count_list[i].driver_non_smoking_signature)," *
            "$(mp.mutation_count_list[i].non_driver_smoking_signature)," *
            "$(mp.mutation_count_list[i].non_driver_non_smoking_signature)" for
            i in 1:mp.index
        ],
        "\n\t",
    )
end

function MutationPhylogeny(total_profiles::Integer)::MutationPhylogeny
    mutation_phylogeny = MutationPhylogeny(
        Vector{MutationalProfile}(undef, total_profiles),
        Vector{MutationCountClass.MutationCount}(undef, total_profiles),
        2, # 1 is reserved for the root profile
        total_profiles,
        0,
    )
    root_profile = MutationalProfile(nothing, 0.0, false, 0, 0)
    mutation_phylogeny.mutational_profile_list[1] = root_profile
    mutation_phylogeny.mutation_count_list[1] = MutationCountClass.MutationCount()
    return mutation_phylogeny
end

function getindex(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
)::MutationalProfile
    return mutation_phylogeny.mutational_profile_list[profile_id-mutation_phylogeny.profiles_printed_to_disk[]]
end

function get_new_profile_id(mutation_phylogeny::MutationPhylogeny)::UInt32
    mutation_phylogeny.index += 1

    if mutation_phylogeny.index > mutation_phylogeny.max_index
        resize!(
            mutation_phylogeny.mutational_profile_list,
            2 * mutation_phylogeny.max_index,
        )
        resize!(
            mutation_phylogeny.mutation_count_list,
            2 * mutation_phylogeny.max_index,
        )
        mutation_phylogeny.max_index *= 2
    end

    return mutation_phylogeny.index + mutation_phylogeny.profiles_printed_to_disk - 1
end

function add_mutation!(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
    cell_id::UInt64,
    mutation_fitness::Float64,
    smoking_signature::Bool,
    step_of_origin::UInt16,
)::UInt32
    if MutationCountClass.total_mutations(
        mutation_phylogeny.mutation_count_list[profile_id],
    ) > 400000
        throw(
            ArgumentError(
                "Too many mutations in cell $cell_id, step $step_of_origin, " *
                "profile $profile_id out of $(mutation_phylogeny.max_index)",
            ),
        )
    end
    mutation_phylogeny.mutational_profile_list[mutation_phylogeny.index] =
        MutationalProfile(
            profile_id,
            mutation_fitness,
            smoking_signature,
            step_of_origin,
            cell_id,
        )
    new_profile_id = get_new_profile_id(mutation_phylogeny)
    mutation_phylogeny.mutation_count_list[new_profile_id] =
        MutationCountClass.copy_incremented(
            mutation_phylogeny.mutation_count_list[profile_id],
            (mutation_fitness >= 0),
            smoking_signature,
        )
    return new_profile_id
end

function mutation_count(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
    driver::Bool,
    smoking_signature::Bool,
)::UInt32
    return MutationCountClass.count(
        mutation_phylogeny.mutation_count_list[profile_id],
        driver,
        smoking_signature,
    )
end

function mutation_count(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
    ::Nothing,
    smoking_signature::Bool,
)::UInt32
    return (
        mutation_count(mutation_phylogeny, profile_id, true, smoking_signature) +
        mutation_count(mutation_phylogeny, profile_id, false, smoking_signature)
    )
end

function mutation_count(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
    driver::Bool,
    ::Nothing,
)::UInt32
    return (
        mutation_count(mutation_phylogeny, profile_id, driver, true) +
        mutation_count(mutation_phylogeny, profile_id, driver, false)
    )
end

function mutation_count(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
    ::Nothing,
    ::Nothing,
)::UInt32
    return (
        mutation_count(mutation_phylogeny, profile_id, true, true) +
        mutation_count(mutation_phylogeny, profile_id, true, false) +
        mutation_count(mutation_phylogeny, profile_id, false, true) +
        mutation_count(mutation_phylogeny, profile_id, false, false)
    )
end

mutation_count(mutation_phylogeny::MutationPhylogeny, profile_id::UInt32)::UInt32 =
    mutation_count(mutation_phylogeny, profile_id, nothing, nothing)

function calculate_mutation_count_from_tree(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
    driver::Union{Bool,Nothing},
    smoking_signature::Union{Bool,Nothing},
)::UInt32
    # TODO: add in those printed to disk
    mutation_count::UInt32 = 0
    while profile_id > 1  # 1 is the root node, has no mutations
        mutation_count += (
            (
                smoking_signature === nothing ||
                mutation_phylogeny[profile_id].smoking_signature == smoking_signature
            ) && (
                driver === nothing ||
                (mutation_phylogeny[profile_id].mutation_fitness >= 0) == driver
            )
        )
        profile_id = mutation_phylogeny[profile_id].parent_profile_id
    end
    return mutation_count
end

function assert_consistent_mutation_counts(
    mutation_phylogeny::MutationPhylogeny,
)::Nothing
    for profile_id in collect(UInt32, 1:mutation_phylogeny.index-1)
        for driver in (true, false)
            for smoking_signature in (true, false)
                @assert mutation_phylogeny.mutational_profile_list[profile_id].parent_profile_id !==
                        nothing || profile_id == 1 "Profile $profile_id of $(mutation_phylogeny.index) has no parent; $(mutation_phylogeny.mutational_profile_list[profile_id])"
                mutation_count_from_tree = calculate_mutation_count_from_tree(
                    mutation_phylogeny,
                    profile_id,
                    driver,
                    smoking_signature,
                )
                mutation_count_from_list = mutation_count(
                    mutation_phylogeny,
                    profile_id,
                    driver,
                    smoking_signature,
                )
                if mutation_count_from_tree != mutation_count_from_list
                    throw(
                        ArgumentError(
                            "Inconsistent mutation counts for profile $profile_id, " *
                            "driver $driver, smoking_signature $smoking_signature: " *
                            "from tree $mutation_count_from_tree, " *
                            "from list $mutation_count_from_list",
                        ),
                    )
                end
            end
        end
    end
    return nothing
end

function get_total_mutation_count(
    mutation_phylogeny::MutationPhylogeny,
    profile_id::UInt32,
)::UInt32
    return (
        mutation_phylogeny.mutation_count_list[profile_id].driver_smoking_signature +
        mutation_phylogeny.mutation_count_list[profile_id].driver_non_smoking_signature +
        mutation_phylogeny.mutation_count_list[profile_id].non_driver_smoking_signature +
        mutation_phylogeny.mutation_count_list[profile_id].non_driver_non_smoking_signature
    )
end

function write_to_csv(
    mutation_phylogeny::MutationPhylogeny,
    this_run_logging_directory::String,
)::Nothing
    directory = joinpath(this_run_logging_directory, "cell_records")
    mkpath(directory)
    open(joinpath(directory, "mutation_phylogeny.csv"), "w") do file
        write(
            file,
            "profile_id,parent_profile_id,mutation_fitness,smoking_signature," *
            "step_of_origin,cell_of_origin\n",
        )
        for profile_id in 1:mutation_phylogeny.index-1
            write(
                file,
                "$profile_id," *
                "$(mutation_phylogeny[profile_id].parent_profile_id)," *
                "$(mutation_phylogeny[profile_id].mutation_fitness)," *
                "$(mutation_phylogeny[profile_id].smoking_signature)," *
                "$(mutation_phylogeny[profile_id].step_of_origin)," *
                "$(mutation_phylogeny[profile_id].cell_of_origin)\n",
            )
        end
    end
    return nothing
end

end
