module CellLatticeClass

import Base: getindex, setindex!

using StatsBase
using ..Records

struct MutationCountLattice
    total_mutations::Array{UInt32,2}
    driver_smoking_signature_mutations::Array{UInt32,2}
    driver_non_smoking_signature_mutations::Array{UInt32,2}
    passenger_smoking_signature_mutations::Array{UInt32,2}
    passenger_non_smoking_signature_mutations::Array{UInt32,2}
end

function MutationCountLattice(grid_side_length::Integer)::MutationCountLattice
    return MutationCountLattice(
        zeros(UInt32, grid_side_length, grid_side_length),
        zeros(UInt32, grid_side_length, grid_side_length),
        zeros(UInt32, grid_side_length, grid_side_length),
        zeros(UInt32, grid_side_length, grid_side_length),
        zeros(UInt32, grid_side_length, grid_side_length),
    )
end

function getindex(mcl::MutationCountLattice, row::Integer, col::Integer)::UInt32
    return mcl.total_mutations[row, col]
end

function replace_location(
    mutation_count_lattice::MutationCountLattice,
    replacing_mutation_count_lattice::MutationCountLattice,
    old_row::Integer,
    old_col::Integer,
    new_row::Integer,
    new_col::Integer,
)::Nothing
    mutation_count_lattice.total_mutations[new_row, new_col] =
        replacing_mutation_count_lattice.total_mutations[old_row, old_col]
    mutation_count_lattice.driver_smoking_signature_mutations[new_row, new_col] =
        replacing_mutation_count_lattice.driver_smoking_signature_mutations[
            old_row,
            old_col,
        ]
    mutation_count_lattice.driver_non_smoking_signature_mutations[new_row, new_col] =
        replacing_mutation_count_lattice.driver_non_smoking_signature_mutations[
            old_row,
            old_col,
        ]
    mutation_count_lattice.passenger_smoking_signature_mutations[new_row, new_col] =
        replacing_mutation_count_lattice.passenger_smoking_signature_mutations[
            old_row,
            old_col,
        ]
    mutation_count_lattice.passenger_non_smoking_signature_mutations[new_row, new_col] =
        replacing_mutation_count_lattice.passenger_non_smoking_signature_mutations[
            old_row,
            old_col,
        ]
    return nothing
end

replace_location(
    mutation_count_lattice::MutationCountLattice,
    old_row::Integer,
    old_col::Integer,
    new_row::Integer,
    new_col::Integer,
)::Nothing = replace_location(
    mutation_count_lattice,
    mutation_count_lattice,
    old_row,
    old_col,
    new_row,
    new_col,
)

replace_location(
    mcl::MutationCountLattice,
    replacing_mcl::MutationCountLattice,
    old_location::Tuple{Integer,Integer},
    new_location::Tuple{Integer,Integer},
)::Nothing = replace_location(
    mcl,
    replacing_mcl,
    old_location[1],
    old_location[2],
    new_location[1],
    new_location[2],
)

replace_location(
    mcl::MutationCountLattice,
    old_location::Tuple{Integer,Integer},
    new_location::Tuple{Integer,Integer},
)::Nothing = replace_location(
    mcl,
    mcl,
    old_location[1],
    old_location[2],
    new_location[1],
    new_location[2],
)

struct CellLattice
    cell_ids::Array{UInt64,2}
    grid_side_length::UInt16
    quiescent_cell_locations::Vector{Tuple{UInt16,UInt16}}
    quiescent_cell_ids::Union{Array{UInt64,2},Nothing}  # because they're in addition to the lattice
    # note quiescent_cell_ids will have size cell_ids / quiescent_spacing (each axis)
    protected_cell_locations::Vector{Tuple{UInt16,UInt16}}
    main_cell_locations::Vector{Tuple{UInt16,UInt16}}
    fitnesses::Array{Float64,2}
    smoking_fitnesses::Union{Array{Float64,2},Nothing}
    quiescent_fitnesses::Union{Array{Float64,2},Nothing} # size grid_side_length / quiescent_spacing (each axis)
    quiescent_smoking_fitnesses::Union{Array{Float64,2},Nothing}
    mutation_count_upper_bound::Ref{Int}
    mutation_counts::MutationCountLattice
    quiescent_mutation_counts::Union{MutationCountLattice,Nothing}
    division_counts::Array{UInt32,2}
    quiescent_division_counts::Union{Array{UInt32,2},Nothing}
    quiescent_mask::BitMatrix
    protected_mask::BitMatrix
    number_of_cells::UInt32
    true_quiescent_gland_fraction::Float64
end

function CellLattice(
    cell_ids::Array{UInt64,2},
    quiescent_cell_locations::Vector{Tuple{UInt16,UInt16}},
    protected_cell_locations::Vector{Tuple{UInt16,UInt16}},
    smoking_driver_fitness_augmentation::Float64,
)::CellLattice
    quiescent_cell_ids = (
        if length(quiescent_cell_locations) == 0
            nothing
        else
            zeros(UInt64, size(cell_ids) .÷ quiescent_cell_locations[1][1])
        end
    )
    main_cell_locations::Vector{Tuple{UInt16,UInt16}} = [
        (x, y) for x in 1:size(cell_ids, 1) for
        y in 1:size(cell_ids, 2) if (x, y) ∉ protected_cell_locations
    ]
    quiescent_mask = get_mask(quiescent_cell_locations, size(cell_ids)[1])
    protected_mask = get_mask(protected_cell_locations, size(cell_ids)[1])
    smoking_fitnesses = (
        if smoking_driver_fitness_augmentation == 0.0
            nothing
        else
            zeros(Float64, size(cell_ids))
        end
    )
    quiescent_fitnesses = (
        if length(quiescent_cell_locations) == 0
            nothing
        else
            zeros(Float64, size(cell_ids) .÷ quiescent_cell_locations[1][1])
        end
    )
    quiescent_smoking_fitnesses = (
        if smoking_driver_fitness_augmentation == 0.0 ||
           length(quiescent_cell_locations) == 0
            nothing
        else
            zeros(Float64, size(cell_ids) .÷ quiescent_cell_locations[1][1])
        end
    )
    mutation_counts = MutationCountLattice(size(cell_ids)[1])
    quiescent_mutation_counts = (
        if length(quiescent_cell_locations) == 0
            nothing
        else
            MutationCountLattice(size(cell_ids)[1] ÷ quiescent_cell_locations[1][1])
        end
    )
    division_counts = zeros(UInt32, size(cell_ids))
    quiescent_division_counts = (
        if length(quiescent_cell_locations) == 0
            nothing
        else
            zeros(UInt32, size(cell_ids) .÷ quiescent_cell_locations[1][1])
        end
    )
    return CellLattice(
        cell_ids,
        size(cell_ids, 1),
        quiescent_cell_locations,
        quiescent_cell_ids,
        protected_cell_locations,
        main_cell_locations,
        zeros(Float64, size(cell_ids)),
        smoking_fitnesses,
        quiescent_fitnesses,
        quiescent_smoking_fitnesses,
        Ref(0),
        mutation_counts,
        quiescent_mutation_counts,
        division_counts,
        quiescent_division_counts,
        quiescent_mask,
        protected_mask,
        size(cell_ids)[1]^2 + length(quiescent_cell_locations),
        length(quiescent_cell_locations) / size(cell_ids)[1]^2,
    )
end

function get_mask(
    locations::Vector{Tuple{UInt16,UInt16}},
    grid_side_length::Integer,
)::BitMatrix
    mask = falses(grid_side_length, grid_side_length)
    for (row, col) in locations
        mask[row, col] = true
    end
    return mask
end

function getindex(cell_lattice::CellLattice, row::UInt16, col::UInt16)::UInt64
    return cell_lattice.cell_ids[row, col]
end

function setindex!(
    cell_lattice::CellLattice,
    cell_id::UInt64,
    row::UInt16,
    col::UInt16,
)::UInt64
    cell_lattice.cell_ids[row, col] = cell_id
    return cell_id
end

function print_lattice(cell_lattice::CellLattice)
    quiescent_spacing = cell_lattice.quiescent_cell_locations[1][1]
    println("printing lattice cell ids; quiescent_spacing = $quiescent_spacing")
    for row in 1:size(cell_lattice.cell_ids, 1)
        for col in 1:size(cell_lattice.cell_ids, 2)
            cell_id = cell_lattice.cell_ids[row, col]
            if cell_id == 0
                # uninitialised
                print("_")
            else
                print(cell_id)
                if (row, col) in cell_lattice.protected_cell_locations
                    print("p")
                end
                if (row, col) in cell_lattice.quiescent_cell_locations
                    print("q")
                    @assert row % quiescent_spacing == 0
                    print(
                        cell_lattice.quiescent_cell_ids[
                            row÷quiescent_spacing,
                            col÷quiescent_spacing,
                        ],
                    )
                else
                    print(" ")
                end
            end
            print(" ")
        end
        println()
    end
    println()
    # also print the quiescent cell ids
    println("printing quiescent cell ids")
    for row in 1:size(cell_lattice.quiescent_cell_ids, 1)
        for col in 1:size(cell_lattice.quiescent_cell_ids, 2)
            cell_id = cell_lattice.quiescent_cell_ids[row, col]
            if cell_id == 0
                print("_")
            else
                print(cell_id)
            end
            print("  ")
        end
        println()
    end
end

function get_neighbour_cells(
    cell_lattice::CellLattice,
    cell_location::Tuple{UInt16,UInt16},
)::Tuple{Vector{UInt64},Vector{Tuple{UInt16,UInt16}},Vector{String}}
    x = cell_location[1]
    y = cell_location[2]
    neighbour_cell_locations::Vector{Tuple{UInt16,UInt16}} =
        get_neighbour_cell_locations(cell_lattice, cell_location)
    neighbour_cell_ids::Vector{UInt64} =
        [cell_lattice[location...] for location in neighbour_cell_locations]
    neighbour_cell_compartment_names::Vector{String} = [
        (
            if cell_lattice.protected_mask[location...]
                "protected"
            else
                "main"
            end
        ) for location in neighbour_cell_locations
    ]

    if cell_lattice.quiescent_mask[x, y]
        push!(
            neighbour_cell_ids,
            cell_lattice.quiescent_cell_ids[
                x÷cell_lattice.quiescent_cell_locations[1][1],
                y÷cell_lattice.quiescent_cell_locations[1][1],
            ],
        )
        push!(neighbour_cell_locations, (x, y))
        push!(neighbour_cell_compartment_names, "quiescent")
    end

    return neighbour_cell_ids,
    neighbour_cell_locations,
    neighbour_cell_compartment_names
end

function get_neighbour_cell_locations(
    cell_lattice::CellLattice,
    cell_location::Tuple{UInt16,UInt16},
)::Vector{Tuple{UInt16,UInt16}}
    x = cell_location[1]
    y = cell_location[2]
    neighbour_cell_locations = Vector{Tuple{UInt16,UInt16}}(undef, 4)
    for neighbour_index in 1:4
        neighbour_x = x + (neighbour_index == 1) - (neighbour_index == 2)
        neighbour_y = y + (neighbour_index == 3) - (neighbour_index == 4)

        if neighbour_x < 1
            neighbour_x = cell_lattice.grid_side_length
        elseif neighbour_x > cell_lattice.grid_side_length
            neighbour_x = UInt16(1)
        end
        if neighbour_y < 1
            neighbour_y = cell_lattice.grid_side_length
        elseif neighbour_y > cell_lattice.grid_side_length
            neighbour_y = UInt16(1)
        end
        neighbour_cell_locations[neighbour_index] = (neighbour_x, neighbour_y)
    end
    return neighbour_cell_locations
end

function get_cell_location(
    cell_lattice::CellLattice,
    cell_location_index::Int,
)::Tuple{UInt16,UInt16}
    # translates 1D index (used for sample() function) to 2D lattice position
    return (
        (cell_location_index - 1) ÷ cell_lattice.grid_side_length + 1,
        (cell_location_index - 1) % cell_lattice.grid_side_length + 1,
    )
end

function get_cell_index(
    cell_lattice::CellLattice,
    cell_location::Tuple{Integer,Integer},
)::Int
    # translates 2D lattice position to 1D index (used for sample() function)
    return (cell_location[1] - 1) * cell_lattice.grid_side_length + cell_location[2]
end

function replace_cell!(
    cell_lattice::CellLattice,
    removed_cell_location::Tuple{UInt16,UInt16},
    dividing_cell_location::Tuple{UInt16,UInt16},
    dividing_cell_quiescent::Bool,
    new_cell_id::UInt64,
)::Nothing
    cell_lattice[removed_cell_location...] = new_cell_id

    if dividing_cell_quiescent
        @assert dividing_cell_location == removed_cell_location (
            "$(dividing_cell_location) != $(removed_cell_location)"
        )
        cell_lattice.fitnesses[removed_cell_location...] =
            cell_lattice.quiescent_fitnesses[(
                dividing_cell_location .÷ cell_lattice.quiescent_cell_locations[1][1]
            )...]
        if cell_lattice.smoking_fitnesses !== nothing
            cell_lattice.smoking_fitnesses[removed_cell_location...] =
                cell_lattice.quiescent_smoking_fitnesses[(
                    dividing_cell_location .÷
                    cell_lattice.quiescent_cell_locations[1][1]
                )...]
        end
        replace_location(
            cell_lattice.mutation_counts,
            cell_lattice.quiescent_mutation_counts,
            dividing_cell_location .÷ cell_lattice.quiescent_cell_locations[1][1],
            removed_cell_location,
        )
    else
        cell_lattice.fitnesses[removed_cell_location...] =
            cell_lattice.fitnesses[dividing_cell_location...]
        if cell_lattice.smoking_fitnesses !== nothing
            cell_lattice.smoking_fitnesses[removed_cell_location...] =
                cell_lattice.smoking_fitnesses[dividing_cell_location...]
        end
        replace_location(
            cell_lattice.mutation_counts,
            dividing_cell_location,
            removed_cell_location,
        )
    end
    return nothing
end

function increment_mutation_count(
    cell_lattice::CellLattice,
    cell_location::Tuple{UInt16,UInt16},
    mutation_count::Int,
    driver_smoking_signature_mutation_count::Int,
    passenger_smoking_signature_mutation_count::Int,
    driver_non_smoking_signature_mutation_count::Int,
    passenger_non_smoking_signature_mutation_count::Int,
    quiescent::Bool,
)::Nothing
    mutation_counts = (
        if quiescent
            cell_lattice.quiescent_mutation_counts
        else
            cell_lattice.mutation_counts
        end
    )
    location = (
        if quiescent
            cell_location .÷ cell_lattice.quiescent_cell_locations[1][1]
        else
            cell_location
        end
    )
    mutation_counts.total_mutations[location...] += mutation_count
    mutation_counts.driver_smoking_signature_mutations[location...] +=
        driver_smoking_signature_mutation_count
    mutation_counts.driver_non_smoking_signature_mutations[location...] +=
        driver_non_smoking_signature_mutation_count
    mutation_counts.passenger_smoking_signature_mutations[location...] +=
        passenger_smoking_signature_mutation_count
    mutation_counts.passenger_non_smoking_signature_mutations[location...] +=
        passenger_non_smoking_signature_mutation_count

    if !quiescent &&
       mutation_counts[location...] > cell_lattice.mutation_count_upper_bound[]
        cell_lattice.mutation_count_upper_bound[] = mutation_counts[location...]
    end
    return nothing
end

function record_mutational_burden(
    cell_lattice::CellLattice,
    mutational_burden_record::Records.Record,
    current_record_number::UInt16,
)::Nothing
    for x in collect(UInt16, 1:cell_lattice.grid_side_length)
        for y in collect(UInt16, 1:cell_lattice.grid_side_length)
            Records.record_mutational_burden!(
                mutational_burden_record,
                current_record_number,
                cell_lattice.cell_ids[x, y],
                cell_lattice.mutation_counts.driver_non_smoking_signature_mutations[
                    x,
                    y,
                ],
                cell_lattice.mutation_counts.driver_smoking_signature_mutations[x, y],
                cell_lattice.mutation_counts.passenger_non_smoking_signature_mutations[
                    x,
                    y,
                ],
                cell_lattice.mutation_counts.passenger_smoking_signature_mutations[
                    x,
                    y,
                ],
                cell_lattice.division_counts[x, y],
                if cell_lattice.protected_mask[x, y]
                    "protected"
                else
                    "main"
                end,
                x,
                y,
            )

            if cell_lattice.quiescent_mask[x, y]
                qui_x, qui_y = (x, y) .÷ cell_lattice.quiescent_cell_locations[1][1]
                Records.record_mutational_burden!(
                    mutational_burden_record,
                    current_record_number,
                    cell_lattice.quiescent_cell_ids[qui_x, qui_y],
                    cell_lattice.quiescent_mutation_counts.driver_non_smoking_signature_mutations[
                        qui_x,
                        qui_y,
                    ],
                    cell_lattice.quiescent_mutation_counts.driver_smoking_signature_mutations[
                        qui_x,
                        qui_y,
                    ],
                    cell_lattice.quiescent_mutation_counts.passenger_non_smoking_signature_mutations[
                        qui_x,
                        qui_y,
                    ],
                    cell_lattice.quiescent_mutation_counts.passenger_smoking_signature_mutations[
                        qui_x,
                        qui_y,
                    ],
                    cell_lattice.quiescent_division_counts[qui_x, qui_y],
                    "quiescent",
                    x,
                    y,
                )
            end
        end
    end
    return nothing
end

function get_alive_cell_ids(
    cell_lattice::CellLattice,
    quiescent_gland_cell_count::Int,
)::Tuple{Vector{UInt64},AbstractWeights{Int}}
    alive_cell_ids = collect(Iterators.flatten(cell_lattice.cell_ids))
    if cell_lattice.quiescent_cell_ids !== nothing
        append!(alive_cell_ids, Iterators.flatten(cell_lattice.quiescent_cell_ids))
        return (
            alive_cell_ids,
            fweights(
                vcat(
                    ones(Int, cell_lattice.grid_side_length^2),
                    quiescent_gland_cell_count *
                    ones(Int, length(cell_lattice.quiescent_cell_ids)),
                ),
            ),
        )
    end
    return alive_cell_ids, uweights(Int, length(alive_cell_ids))
end

function get_mutation_counts_list(
    cell_lattice::CellLattice,
    quiescent::Bool,
    protected::Bool,
    driver::Union{Bool,Nothing},
    smoking_signature::Union{Bool,Nothing},
)::Vector{Int}
    mutation_counts_list = Vector{Int}()
    mutation_counts_lattice = (
        if quiescent
            cell_lattice.quiescent_mutation_counts
        else
            cell_lattice.mutation_counts
        end
    )
    lattice_side_length = size(mutation_counts_lattice.total_mutations, 1)
    for driver in (driver === nothing ? [true, false] : [driver])
        for smoking_signature in
            (smoking_signature === nothing ? [true, false] : [smoking_signature])
            for x in collect(UInt16, 1:lattice_side_length)
                for y in collect(UInt16, 1:lattice_side_length)
                    if !quiescent && protected && !cell_lattice.protected_mask[x, y]
                        continue
                    end

                    if driver
                        if smoking_signature
                            push!(
                                mutation_counts_list,
                                mutation_counts_lattice.driver_smoking_signature_mutations[
                                    x,
                                    y,
                                ],
                            )
                        else
                            push!(
                                mutation_counts_list,
                                mutation_counts_lattice.driver_non_smoking_signature_mutations[
                                    x,
                                    y,
                                ],
                            )
                        end
                    else
                        if smoking_signature
                            push!(
                                mutation_counts_list,
                                mutation_counts_lattice.passenger_smoking_signature_mutations[
                                    x,
                                    y,
                                ],
                            )
                        else
                            push!(
                                mutation_counts_list,
                                mutation_counts_lattice.passenger_non_smoking_signature_mutations[
                                    x,
                                    y,
                                ],
                            )
                        end
                    end
                end
            end
        end
    end
    return mutation_counts_list
end

function calculate_fitness_difference(
    cell_lattice::CellLattice,
    dividing_cell_location::Tuple{UInt16,UInt16},
    quiescent_cell_dividing::Bool,
    removed_cell_location::Tuple{UInt16,UInt16},
    smoking::Bool,
)::Float64
    use_smoking_fitnesses = smoking && cell_lattice.smoking_fitnesses !== nothing
    removed_cell_fitness = (
        if use_smoking_fitnesses
            cell_lattice.smoking_fitnesses[removed_cell_location...]
        else
            cell_lattice.fitnesses[removed_cell_location...]
        end
    )
    new_cell_fitness = (
        if use_smoking_fitnesses
            if quiescent_cell_dividing
                cell_lattice.quiescent_smoking_fitnesses[(
                    dividing_cell_location .÷
                    cell_lattice.quiescent_cell_locations[1][1]
                )...]
            else
                cell_lattice.smoking_fitnesses[dividing_cell_location...]
            end
        else
            if quiescent_cell_dividing
                cell_lattice.quiescent_fitnesses[(
                    dividing_cell_location .÷
                    cell_lattice.quiescent_cell_locations[1][1]
                )...]
            else
                cell_lattice.fitnesses[dividing_cell_location...]
            end
        end
    )
    return (new_cell_fitness - removed_cell_fitness) / cell_lattice.grid_side_length^2
end

end
