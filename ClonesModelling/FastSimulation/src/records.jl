module Records

struct Record
    names::Vector{Symbol}
    data::Vector{Vector{Any}}
    row_index::Ref{Int}
    max_rows::Ref{Int}
end

function Record(colnames::Vector{Pair{Symbol,DataType}}, initial_max_rows::Int)::Record
    names = [colname for (colname, _) in colnames]
    data = Vector{Vector{Any}}([
        Vector{type}(undef, initial_max_rows) for (_, type) in colnames
    ])
    return Record(names, data, Ref(1), Ref(initial_max_rows))
end

function add_record!(record::Record, values::Tuple)::Nothing
    index = record.row_index[]

    ensure_index_in_range!(record)

    for (i, value) in enumerate(values)
        record.data[i][index] = value
    end

    record.row_index[] += 1
    return nothing
end

function ensure_index_in_range!(record::Record)::Nothing
    index = record.row_index[]
    max_rows = record.max_rows[]
    if index > max_rows
        new_capacity = ceil(Int, 1.3 * max_rows)
        for data_list in record.data
            resize!(data_list, new_capacity)
        end
        record.max_rows[] = new_capacity
    end
    return nothing
end

function write_to_csv(
    record::Record,
    filename::String,
    include_titles::Bool = true,
)::Nothing
    if include_titles
        open(filename, "w") do file
            return println(file, join(record.names, ","))
        end
    end

    column_count = length(record.names)
    open(filename, "a") do file
        for i in 1:record.row_index[]-1
            for j in 1:column_count-1
                print(file, record.data[j][i])
                print(file, ",")
            end
            print(file, record.data[column_count][i])
            print(file, "\n")
        end
    end
end

@enum CompartmentName begin
    total
    main
    quiescent
    protected
    quiescent_main
    quiescent_protected
end

function get_compartment(compartment_name::String)::CompartmentName
    if compartment_name == "total"
        return total
    elseif compartment_name == "main"
        return main
    elseif compartment_name == "quiescent"
        return quiescent
    elseif compartment_name == "protected"
        return protected
    elseif compartment_name == "quiescent_main"
        return quiescent_main
    elseif compartment_name == "quiescent_protected"
        return quiescent_protected
    else
        throw(ArgumentError("Invalid compartment name: $compartment_name"))
    end
end

function PatientSimulationRecord(age::Int)::Record
    return Record(
        [
            :year => Int,
            :compartment => CompartmentName,
            :cell_count => Int,
            :new_cell_count => Int,
            :differentiated_cell_count => Int,
            :immune_death_count => Int,
        ],
        age,
    )
end

function add_yearly_record!(
    patient_simulation_record::Record,
    year::UInt8,
    compartment::String,
    cell_count::Int,
    new_cell_count::Int,
    differentiated_cell_count::Int,
    immune_death_count::Int,
)::Nothing
    add_record!(
        patient_simulation_record,
        (
            year,
            get_compartment(compartment),
            cell_count,
            new_cell_count,
            differentiated_cell_count,
            immune_death_count,
        ),
    )
    return nothing
end

function MutationalBurdenRecord(total_records::Int)::Record
    return Record(
        [
            :step_number => UInt16,
            :cell_id => UInt64,
            :driver_non_smoking_signature_mutations => UInt32,
            :driver_smoking_signature_mutations => UInt32,
            :passenger_non_smoking_signature_mutations => UInt32,
            :passenger_smoking_signature_mutations => UInt32,
            :divisions => UInt32,
            :compartment => CompartmentName,
        ],
        total_records,
    )
end

function record_mutational_burden!(
    mutational_burden_record::Record,
    step_number::UInt16,
    cell_id::UInt64,
    driver_non_smoking_signature_mutations::UInt32,
    driver_smoking_signature_mutations::UInt32,
    passenger_non_smoking_signature_mutations::UInt32,
    passenger_smoking_signature_mutations::UInt32,
    divisions::UInt32,
    compartment::String,
)::Nothing
    # to avoid iterating, we manually reimplement for this specific case
    index = mutational_burden_record.row_index[]
    ensure_index_in_range!(mutational_burden_record)

    mutational_burden_record.data[1][index] = step_number
    mutational_burden_record.data[2][index] = cell_id
    mutational_burden_record.data[3][index] = driver_non_smoking_signature_mutations
    mutational_burden_record.data[4][index] = driver_smoking_signature_mutations
    mutational_burden_record.data[5][index] = passenger_non_smoking_signature_mutations
    mutational_burden_record.data[6][index] = passenger_smoking_signature_mutations
    mutational_burden_record.data[7][index] = divisions
    mutational_burden_record.data[8][index] = get_compartment(compartment)

    mutational_burden_record.row_index[] += 1
    return nothing
end

function SpatialMutationalBurdenRecord(total_records::Int)::Record
    return Record(
        [
            :record_number => UInt16,
            :cell_id => UInt64,
            :driver_non_smoking_signature_mutations => UInt32,
            :driver_smoking_signature_mutations => UInt32,
            :passenger_non_smoking_signature_mutations => UInt32,
            :passenger_smoking_signature_mutations => UInt32,
            :divisions => UInt32,
            :compartment => CompartmentName,
            :x => UInt16,
            :y => UInt16,
        ],
        total_records,
    )
end

function record_mutational_burden!(
    mutational_burden_record::Record,
    record_number::UInt16,
    cell_id::UInt64,
    driver_non_smoking_signature_mutations::UInt32,
    driver_smoking_signature_mutations::UInt32,
    passenger_non_smoking_signature_mutations::UInt32,
    passenger_smoking_signature_mutations::UInt32,
    divisions::UInt32,
    compartment::String,
    x::UInt16,
    y::UInt16,
)::Nothing
    # to avoid iterating, we manually reimplement for this specific case
    index = mutational_burden_record.row_index[]
    ensure_index_in_range!(mutational_burden_record)

    mutational_burden_record.data[1][index] = record_number
    mutational_burden_record.data[2][index] = cell_id
    mutational_burden_record.data[3][index] = driver_non_smoking_signature_mutations
    mutational_burden_record.data[4][index] = driver_smoking_signature_mutations
    mutational_burden_record.data[5][index] = passenger_non_smoking_signature_mutations
    mutational_burden_record.data[6][index] = passenger_smoking_signature_mutations
    mutational_burden_record.data[7][index] = divisions
    mutational_burden_record.data[8][index] = get_compartment(compartment)
    mutational_burden_record.data[9][index] = x
    mutational_burden_record.data[10][index] = y

    mutational_burden_record.row_index[] += 1
    return nothing
end

function write_mutational_burden_record(
    mutational_burden_record::Record,
    this_run_logging_directory::String,
    patient_id::String,
    include_titles::Bool,
)::Nothing
    mutational_burden_directory = "$this_run_logging_directory/cell_records/mutational_burden"
    mkpath(mutational_burden_directory)
    mutational_burden_filename = "$mutational_burden_directory/$patient_id.csv"
    write_to_csv(mutational_burden_record, mutational_burden_filename, include_titles)
    return nothing
end

function FitnessRecord(
    age::Float64,
    record_frequency::Int,
    compartment_count::Int,
)::Record
    total_records = ceil(Int, (age + 1) * record_frequency * (compartment_count + 1))
    return Record(
        [
            :step_number => UInt16,
            :compartment => CompartmentName,
            :sm_mean_fitness => Float64,
            :ns_mean_fitness => Float64,
            :sm_std_fitness => Float64,
            :ns_std_fitness => Float64,
            :normalisation_constant => Float64,
        ],
        total_records,
    )
end

function record_fitness_summary(
    fitness_record::Record,
    step_number::UInt16,
    compartment::String,
    sm_mean_fitness::Float64,
    ns_mean_fitness::Float64,
    sm_std_fitness::Float64,
    ns_std_fitness::Float64,
    normalisation_constant::Float64,
)::Nothing
    add_record!(
        fitness_record,
        (
            step_number,
            get_compartment(compartment),
            sm_mean_fitness,
            ns_mean_fitness,
            sm_std_fitness,
            ns_std_fitness,
            normalisation_constant,
        ),
    )
    return nothing
end

function write_fitness_summaries(
    fitness_record::Record,
    this_run_logging_directory::String,
    patient_id::String,
    include_titles::Bool = true,
)::Nothing
    fitness_record_directory = "$this_run_logging_directory/cell_records/fitness_summaries"
    mkpath(fitness_record_directory)
    write_to_csv(
        fitness_record,
        "$fitness_record_directory/$patient_id.csv",
        include_titles,
    )
    return nothing
end

end
