module SmokingRecordClass
using CSV
using DataFrames

@enum SmokingStatus begin
    never
    ex
    current
end

struct SmokingRecord
    patient::String
    status::SmokingStatus
    age::Float64
    start_smoking_age::Union{Float64,Nothing}
    stop_smoking_age::Union{Float64,Nothing}
    pack_years::Union{Float64,Nothing}
    true_data_cell_count::Union{UInt16,Nothing}
end

function get_smoker_record(
    patient::String,
    age::Union{Float64,Int},
    start_smoking_age::Union{Float64,Int},
    pack_years::Float64,
    true_data_cell_count::Union{Integer,Nothing} = nothing,
)::SmokingRecord
    return SmokingRecord(
        patient,
        current,
        Float64(age),
        Float64(start_smoking_age),
        nothing,
        pack_years,
        (true_data_cell_count === nothing) ? nothing : UInt16(true_data_cell_count),
    )
end

function get_ex_smoker_record(
    patient::String,
    age::Union{Float64,Int},
    start_smoking_age::Union{Float64,Int},
    stop_smoking_age::Union{Float64,Int},
    pack_years::Float64,
    true_data_cell_count::Union{Integer,Nothing} = nothing,
)::SmokingRecord
    return SmokingRecord(
        patient,
        ex,
        Float64(age),
        Float64(start_smoking_age),
        Float64(stop_smoking_age),
        pack_years,
        (true_data_cell_count === nothing) ? nothing : UInt16(true_data_cell_count),
    )
end

function get_never_smoker_record(
    patient::String,
    age::Union{Float64,Int},
    true_data_cell_count::Union{Integer,Nothing} = nothing,
)::SmokingRecord
    return SmokingRecord(
        patient,
        never,
        Float64(age),
        nothing,
        nothing,
        0.0,
        (true_data_cell_count === nothing) ? nothing : UInt16(true_data_cell_count),
    )
end

function smoking_at_age(smoking_record::SmokingRecord, age::Float64)::Bool
    if smoking_record.status == never
        return false
    elseif smoking_record.status == ex
        return age >= smoking_record.start_smoking_age &&
               age <= smoking_record.stop_smoking_age
    elseif smoking_record.status == current
        return age >= smoking_record.start_smoking_age
    end
end

function read_smoking_records(
    include_infants::Bool,
    first_patient_test::Bool,
    status_representative_test::Bool,
    exclude_nature_genetics::Bool,
    supersample_patient_cohort::Bool,
    epidemiology_test_cohort::Bool,
)::Vector{SmokingRecord}
    data_dir = (
        if isdir("ClonesModelling/data")
            "ClonesModelling/data/patient_data"
        else
            "/cluster/project2/clones_modelling/ClonesModelling/ClonesModelling/"
            "data/patient_data"
        end
    )
    sr_dataframe = CSV.read(
        joinpath(data_dir, if supersample_patient_cohort
            "supersample_smoking_records.csv"
        else
            if epidemiology_test_cohort
                "epidemiology_test_smoking_records.csv"
            else
                "smoking_records.csv"
            end
        end),
        DataFrame,
    )
    if exclude_nature_genetics
        sr_dataframe = sr_dataframe[.!sr_dataframe.nature_genetics, :]
    end
    if !include_infants
        sr_dataframe = sr_dataframe[sr_dataframe.age.>=5, :]
    end
    if first_patient_test & status_representative_test
        println(
            "Warning: first_patient_test and status_representative_test are both " *
            "true, status_representative_test will be ignored",
        )
    end
    if first_patient_test
        sr_dataframe = sr_dataframe[1:1, :]
    elseif status_representative_test
        return [
            get_smoker_record("test_smoker", 80, 20, 60.0, 25),
            get_ex_smoker_record("test_ex_smoker", 80, 20, 60, 40.0, 25),
            get_never_smoker_record("test_never_smoker", 80.0, 25),
        ]
    end
    if !hasproperty(sr_dataframe, :n_cells)
        sr_dataframe.n_cells = fill(nothing, nrow(sr_dataframe))
    end

    smoking_records = Vector{SmokingRecord}()
    for row in eachrow(sr_dataframe)
        if row.smoking_status == "non-smoker"
            push!(
                smoking_records,
                get_never_smoker_record(String(row.patient), row.age, row.n_cells),
            )
        elseif row.smoking_status == "ex-smoker"
            push!(
                smoking_records,
                get_ex_smoker_record(
                    String(row.patient),
                    row.age,
                    row.start_smoking_age,
                    row.stop_smoking_age,
                    row.pack_years === missing ? 0.0 : row.pack_years,
                    row.n_cells,
                ),
            )
        elseif row.smoking_status == "smoker"
            push!(
                smoking_records,
                get_smoker_record(
                    String(row.patient),
                    row.age,
                    row.start_smoking_age,
                    row.pack_years === missing ? 0.0 : row.pack_years,
                    row.n_cells,
                ),
            )
        end
    end
    return smoking_records
end

function years_smoking(smoking_record::SmokingRecord)::Float64
    if smoking_record.status == never
        return 0.0
    elseif smoking_record.status == ex
        return smoking_record.stop_smoking_age - smoking_record.start_smoking_age
    elseif smoking_record.status == current
        return smoking_record.age - smoking_record.start_smoking_age
    end
end

function years_not_smoking(smoking_record::SmokingRecord)::Float64
    if smoking_record.status == never
        return smoking_record.age
    elseif smoking_record.status == ex
        return (
            smoking_record.age - smoking_record.stop_smoking_age +
            smoking_record.start_smoking_age
        )
    elseif smoking_record.status == current
        return smoking_record.start_smoking_age
    end
end
end
