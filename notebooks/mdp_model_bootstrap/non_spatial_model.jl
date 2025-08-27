using Roots

function prob_extinction_by_t(t::Int, F::Float64)::Float64
    return (1 - F) * (exp(F * t) - 1) / ((1 + F) * exp(F * t) - (1 - F))
end

function expected_clone_size_at_t(t::Int, F::Float64)::Float64
    return ((1 + F) * exp(F * t) - (1 - F)) / (2F)
end

M = 60
N = 100

# solve the value of F for which expected_clone_size at M is N 
F = find_zero(
    F -> expected_clone_size_at_t(M, F) * (1 - prob_extinction_by_t(M, F)) - N,
    0.1,
)
println(
    "non-spatial simulation reaches $N after $M years for F>$F \n" *
    " prob extinction by $M years is $(100*prob_extinction_by_t(M, F))%",
)
