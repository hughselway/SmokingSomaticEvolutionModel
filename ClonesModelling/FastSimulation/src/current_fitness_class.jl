module CurrentFitnessClass

using Random

function get_fitness_change(
    randomiser::Random.Xoshiro,
    fitness_change_scale::Float64,
    mutation_driver_probability::Float64,
    fitness_change_probability::Float64,
)::Float64
    if rand(randomiser) > fitness_change_probability
        return 0.0
    end
    # Exponential with scale fitness_change_scale
    driver = rand(randomiser) < mutation_driver_probability
    return (driver ? 1 : -1) * fitness_change_scale * -log(rand(randomiser))
end

mutable struct CurrentFitness
    smoking::Float64
    non_smoking::Float64
end

function get_value(cf::CurrentFitness, smoking_status::Bool)::Float64
    return smoking_status ? cf.smoking : cf.non_smoking
end

function add!(cf::CurrentFitness, adjustment_value::Float64)::Nothing
    cf.smoking += adjustment_value
    cf.non_smoking += adjustment_value
    return nothing
end

function subtract!(cf::CurrentFitness, adjustment_value::Float64)::Nothing
    cf.smoking -= adjustment_value
    cf.non_smoking -= adjustment_value
    return nothing
end

function apply_fitness_change!(
    cf::CurrentFitness,
    fitness_change::Float64,
    smoking_driver_fitness_augmentation::Float64,
    protection_coefficient::Float64,
)::Nothing
    cf.smoking += get_smoking_fitness_change(
        fitness_change,
        smoking_driver_fitness_augmentation,
        protection_coefficient,
    )
    cf.non_smoking += fitness_change
    return nothing
end

function get_smoking_fitness_change(
    fitness_change::Float64,
    smoking_driver_fitness_augmentation::Float64,
    protection_coefficient::Float64,
)::Float64
    return fitness_change * (
        1 +
        protection_coefficient *
        smoking_driver_fitness_augmentation *
        (fitness_change > 0)
    )
end

copy(cf::CurrentFitness) = CurrentFitness(cf.smoking, cf.non_smoking)

function get_projected_fitness(
    cf::CurrentFitness,
    symmetric_division_prob::Float64,
    smoking::Bool,
    current_normalisation_constant::Float64,
)
    fitness_value = get_value(cf, smoking) - current_normalisation_constant
    return project_fitness_value(fitness_value, symmetric_division_prob)
end

function get_projected_fitness(
    cf::CurrentFitness,
    symmetric_division_prob::Float64,
    smoking::Bool,
)
    fitness_value = get_value(cf, smoking)
    return project_fitness_value(fitness_value, symmetric_division_prob)
end

function project_fitness_value(
    fitness_value::Float64,
    symmetric_division_prob::Float64,
)::Float64
    if fitness_value > 700
        # exp(700) is ~the largest value that can be represented as a Float64
        return symmetric_division_prob
    elseif fitness_value < -700
        return -symmetric_division_prob
    else
        return (2 * symmetric_division_prob) / (1 + exp(-fitness_value)) -
               symmetric_division_prob
    end
end

end
