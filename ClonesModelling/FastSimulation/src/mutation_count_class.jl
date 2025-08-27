module MutationCountClass

struct MutationCount
    driver_smoking_signature::UInt32
    driver_non_smoking_signature::UInt32
    non_driver_smoking_signature::UInt32
    non_driver_non_smoking_signature::UInt32
end

MutationCount() = MutationCount(0, 0, 0, 0)

function count(mc::MutationCount, driver::Bool, smoking_signature::Bool)::UInt32
    if driver
        if smoking_signature
            return mc.driver_smoking_signature
        else
            return mc.driver_non_smoking_signature
        end
    else
        if smoking_signature
            return mc.non_driver_smoking_signature
        else
            return mc.non_driver_non_smoking_signature
        end
    end
end

function total_mutations(mc::MutationCount)::UInt32
    return mc.driver_smoking_signature +
           mc.driver_non_smoking_signature +
           mc.non_driver_smoking_signature +
           mc.non_driver_non_smoking_signature
end

function copy_incremented(
    mc::MutationCount,
    driver::Bool,
    smoking_signature::Bool,
)::MutationCount
    return MutationCount(
        mc.driver_smoking_signature + (driver && smoking_signature),
        mc.driver_non_smoking_signature + (driver && !smoking_signature),
        mc.non_driver_smoking_signature + (!driver && smoking_signature),
        mc.non_driver_non_smoking_signature + (!driver && !smoking_signature),
    )
end

end
