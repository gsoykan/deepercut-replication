import CUDA
using Knet
using Images
using Distributed

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

function simple_mse(x, y)
    sum(abs2, (x - y)) / size(y)[end]
end

function clear_gpu_memory()
    CUDA.reclaim()
    CUDA.memory_status()
    GC.gc(true)
end

function substract_mean_img_from_data!(data, mean_pixel)
    step_size = 2000
    data_length = length(data)
    for i = 1:step_size:data_length
        upper_bound = min(i + step_size, data_length)
        data.x[i:upper_bound] = map(
            element -> substract_mean_img_from_data_for_element(element, mean_pixel),
            enumerate(data.x[i:upper_bound]),
        )
    end
end

function substract_mean_img_from_data_for_element(enumerated_img, mean_pixel)
    x = enumerated_img[2]
    count = enumerated_img[1]
    if count % 1024 == 1
        println(count)
    end
    return x .- mean_pixel
end
