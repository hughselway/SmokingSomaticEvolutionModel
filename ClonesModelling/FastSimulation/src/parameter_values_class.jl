module ParameterValuesClass

struct QuiescentParameters
    quiescent_fraction::Float64
    quiescent_divisions_per_year::Float64
    ambient_quiescent_divisions_per_year::Float64
    quiescent_gland_cell_count::Union{Int,Nothing}
end

function QuiescentParameters(
    quiescent_fraction,
    quiescent_divisions_per_year,
)::Union{QuiescentParameters,Nothing}
    if quiescent_fraction > 0
        return QuiescentParameters(
            quiescent_fraction,
            quiescent_divisions_per_year,
            0.0,
            nothing,
        )
    else
        return nothing
    end
end

struct QuiescentProtectedParameters
    quiescent_protection_coefficient::Float64
end

struct ProtectedParameters
    protected_fraction::Float64
    protection_coefficient::Float64
    protected_region_radius::Union{Int,Nothing}
end

function ProtectedParameters(protected_fraction, protection_coefficient)
    return ProtectedParameters(protected_fraction, protection_coefficient, nothing)
end

struct ImmuneResponseParameters
    immune_death_rate::Float64
    smoking_immune_coeff::Float64
end

struct SmokingDriverParameters
    smoking_driver_fitness_augmentation::Float64
end

struct ParameterValues
    smoking_mutations_per_year::Float64
    non_smoking_mutations_per_year::Float64
    fitness_change_scale::Float64
    fitness_change_probability::Float64
    smoking_divisions_per_year::Union{Float64,Nothing}
    non_smoking_divisions_per_year::Union{Float64,Nothing}
    mutation_rate_multiplier_shape::Union{Float64,Nothing}
    quiescent_parameters::Union{Nothing,QuiescentParameters}
    quiescent_protected_parameters::Union{Nothing,QuiescentProtectedParameters}
    protected_parameters::Union{Nothing,ProtectedParameters}
    immune_response_parameters::Union{Nothing,ImmuneResponseParameters}
    smoking_driver_parameters::Union{Nothing,SmokingDriverParameters}
end

function ParameterValues(
    smoking_mutation_rate_augmentation::Float64,
    non_smoking_mutations_per_year::Float64,
    fitness_change_scale::Union{Int,Float64},
    fitness_change_probability::Float64,
    smoking_division_rate_augmentation::Union{Float64,Nothing},
    non_smoking_divisions_per_year::Union{Float64,Nothing},
    mutation_rate_multiplier_shape::Union{Float64,Nothing},
    quiescent_fraction::Union{Float64,Nothing},
    quiescent_divisions_per_year::Union{Float64,Nothing},
    ambient_quiescent_divisions_per_year::Union{Float64,Nothing},
    quiescent_gland_cell_count::Union{Int,Nothing},
    quiescent_protection_coefficient::Union{Float64,Nothing},
    protected_fraction::Union{Float64,Nothing},
    protection_coefficient::Union{Float64,Nothing},
    protected_region_radius::Union{Int,Nothing},
    immune_death_rate::Union{Float64,Nothing},
    smoking_immune_coeff::Union{Float64,Nothing},
    smoking_driver_fitness_augmentation::Union{Float64,Nothing},
)
    quiescent_parameters = (
        if (
            quiescent_fraction !== nothing &&
            quiescent_divisions_per_year !== nothing &&
            quiescent_fraction > 0
        )
            QuiescentParameters(
                quiescent_fraction,
                quiescent_divisions_per_year,
                ambient_quiescent_divisions_per_year,
                quiescent_gland_cell_count,
            )
        else
            nothing
        end
    )
    quiescent_protected_parameters = (
        if quiescent_protection_coefficient !== nothing
            QuiescentProtectedParameters(quiescent_protection_coefficient)
        else
            nothing
        end
    )
    protected_parameters = (
        if (
            protected_fraction !== nothing &&
            protection_coefficient !== nothing &&
            protected_fraction > 0
        )
            ProtectedParameters(
                protected_fraction,
                protection_coefficient,
                protected_region_radius,
            )
        else
            nothing
        end
    )
    immune_response_parameters = (
        if immune_death_rate !== nothing && smoking_immune_coeff !== nothing
            ImmuneResponseParameters(immune_death_rate, smoking_immune_coeff)
        else
            nothing
        end
    )
    smoking_driver_parameters = (
        if smoking_driver_fitness_augmentation !== nothing
            SmokingDriverParameters(smoking_driver_fitness_augmentation)
        else
            nothing
        end
    )
    return ParameterValues(
        non_smoking_mutations_per_year * (1 + smoking_mutation_rate_augmentation),
        non_smoking_mutations_per_year,
        Float64(fitness_change_scale),
        fitness_change_probability,
        non_smoking_divisions_per_year * (1 + smoking_division_rate_augmentation),
        non_smoking_divisions_per_year,
        mutation_rate_multiplier_shape,
        quiescent_parameters,
        quiescent_protected_parameters,
        protected_parameters,
        immune_response_parameters,
        smoking_driver_parameters,
    )
end

function is_active(parameter_values::ParameterValues, hypothesis_module::Symbol)::Bool
    if hypothesis_module == :Quiescent
        return parameter_values.quiescent_parameters !== nothing
    elseif hypothesis_module == :QuiescentProtected
        return parameter_values.quiescent_protected_parameters !== nothing
    elseif hypothesis_module == :Protected
        return parameter_values.protected_parameters !== nothing
    elseif hypothesis_module == :ImmuneResponse
        return parameter_values.immune_response_parameters !== nothing
    elseif hypothesis_module == :SmokingDriver
        return parameter_values.smoking_driver_parameters !== nothing
    else
        return false
    end
end

function non_smoking_divisions_per_year(parameter_values::ParameterValues)::Float64
    if parameter_values.non_smoking_divisions_per_year !== nothing
        return parameter_values.non_smoking_divisions_per_year
    else
        return 33.0
    end
end

function smoking_divisions_per_year(parameter_values::ParameterValues)::Float64
    if parameter_values.smoking_divisions_per_year !== nothing
        return parameter_values.smoking_divisions_per_year
    else
        return 59.4
    end
end

function get_mutation_driver_probability(
    fitness_change_scale::Float64,
    fitness_change_probability::Float64,
    spatial::Bool,
)::Float64
    # as calculated from bootstrap simplified model
    significance_threshold = spatial ? 0.110 : 0.0310
    prob_significant_driver_given_positive_fitness =
        (
            (1 - significance_threshold) / (1 + significance_threshold)
        )^(1 / fitness_change_scale) # derived from exponential distribution
    # CDC driver genes / protein-coding genes
    empirical_driver_frequency_in_exome = 719 / 20000
    protein_coding_fraction_of_genome = 0.0174
    driver_gene_fraction_in_genome =
        empirical_driver_frequency_in_exome * protein_coding_fraction_of_genome
    non_synonymous_mutation_probability = 0.77 # from local analysis
    driver_gene_driver_mutation_probability = non_synonymous_mutation_probability * 0.49 # from https://doi.org/10.1016/j.cell.2017.09.042
    return driver_gene_fraction_in_genome * driver_gene_driver_mutation_probability /
           (prob_significant_driver_given_positive_fitness * fitness_change_probability)
end

end
