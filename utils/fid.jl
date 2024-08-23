using Statistics
using LinearAlgebra: tr

function fid_score(fake_data, real_data)

    μ_fake,  σ_fake = ( mean(fake_data, dims=(2)), cov(fake_data, dims=2) )
    μ_real , σ_real = ( mean(real_data, dims=(2)), cov(real_data, dims=2) )

    covsqrt = real.(sqrt(σ_fake * σ_real))


    fid = norm(μ_fake - μ_real)^2 + tr(σ_fake + σ_real - 2 * covsqrt)
end

function ind_var_scores(fake_data)

    μ = mean(fake_data, dims=2)

    vars = mapslices(x -> (μ .- x).^2, fake_data, dims=(1))

    return sum(vars, dims=1)
end

function make_cdf(fake_data)

    vars = ind_var_scores(fake_data)

    sort!(vars, dims=2)

    prob = range(0,1, length(vars))

    return vars, prob
end

function make_pdf(fake_data)

    vars = ind_var_scores(fake_data)

    sort!(vars, dims=2)

    return vars
end