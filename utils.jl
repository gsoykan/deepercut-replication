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