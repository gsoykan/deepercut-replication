using Knet

function compute_loss_for_sets(model, named_x_y_tuples)
    loss_dict = Dict()
    for named_x_y_tuple in named_x_y_tuples
        name = named_x_y_tuple[1]
        x = named_x_y_tuple[2]
        y = named_x_y_tuple[3]
        loss_dict["$name-loss"] = model(x, y)
    end
    return loss_dict
end

function compute_accuracy_for_sets(model, named_x_y_tuples)
    accuracy_dict = Dict()
    for named_x_y_tuple in named_x_y_tuples
        name = named_x_y_tuple[1]
        x = named_x_y_tuple[2]
        y = named_x_y_tuple[3]
        current_res = model(x)
        accuracy_dict["$name-accuracy-mae"] = mae(current_res, y)
    end
    return accuracy_dict
end

function compute_mae_for_data(model, data)
    sum = 0
    count = 0
    for (x, y) in data 
        current_res = model(x)
        sum += simple_mae(current_res, y)
        count += 1
    end
    return sum / count
end

function simple_mae(x, y)
    sum(abs, (x - y)) / size(y)[end]
end

# TODO: convert this to modular layer!!!? (You may ask TAs)
# Batch Normalization Layer
# works both for convolutional and fully connected layers
# mode, 0=>train, 1=>test
function batchnorm_as_function(w, x, ms; training=AutoGrad.recording(), epsilon=1e-5)
    mu, sigma = nothing, nothing
    if training 
        d = ndims(x) == 4 ? (1, 2, 4) : (2,)
        s = prod(size(x, d...))
        mu = sum(x, d) / s
        x0 = x .- mu
        x1 = x0 .* x0
        sigma = sqrt(epsilon + (sum(x1, d)) / s)
    else
        mu = popfirst!(ms)
        sigma = popfirst!(ms)
    end

    # we need getval in backpropagation
    push!(ms, AutoGrad.value(mu), AutoGrad.value(sigma))
    xhat = (x .- mu) ./ sigma
    return w[1] .* xhat .+ w[2]
end