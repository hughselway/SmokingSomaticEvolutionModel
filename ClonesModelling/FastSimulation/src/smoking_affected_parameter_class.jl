module SmokingAffectedParameterClass

struct SmokingAffectedParameter
    smoking_value::Float64
    non_smoking_value::Float64
end

get_value(sap::SmokingAffectedParameter, smoking_status::Bool)::Float64 =
    smoking_status ? sap.smoking_value : sap.non_smoking_value

get_protected_value(
    sap::SmokingAffectedParameter,
    smoking_status::Bool,
    protection_coefficient::Float64,
)::Float64 =
    sap.non_smoking_value +
    (get_value(sap, smoking_status) - sap.non_smoking_value) * protection_coefficient

end
