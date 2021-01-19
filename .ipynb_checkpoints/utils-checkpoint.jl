import CUDA
using Knet
using Images
using Distributed
include("./deeper-cut/deeper-cut.config.jl")
include("./deeper-cut/mpii.annotation.reader.jl")
using DataFrames
using CSV

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

function substract_mean_img_from_data!(data, mean_pixel, atype)
    step_size = 2000
    data_length = length(data)
    for i = 1:step_size:data_length
        upper_bound = min(i + step_size, data_length)
        data.x[i:upper_bound] = map(
            element ->
                substract_mean_img_from_data_for_element(element, mean_pixel, atype),
            enumerate(data.x[i:upper_bound]),
        )
    end
end

function substract_mean_img_from_data_for_element(enumerated_img, mean_pixel, atype)
    x = enumerated_img[2]
    count = enumerated_img[1]
    if count % 1024 == 1
        println(count)
    end
    result = Array(x) .- mean_pixel
    return atype(result)
end

function mirror_variant_data(variant_data)
    new_x = deepcopy(variant_data.x)
    new_y = deepcopy(variant_data.y)
    counter = 1
    original_shuffle = variant_data.shuffle
    variant_data.shuffle = false
    for (x, y) in variant_data
        arr_x = Array(x)
        arr_y = Array(y)
        arr_x = reduce_dim(arr_x)
        arr_y = reduce_dim(arr_y)
        flipped_x = fliplr3D(arr_x)
        flipped_y = exchange_symmetric_joint_ids_for_flipped_labels(fliplr3D(arr_y))
        flipped_x = add_dim(flipped_x) |> variant_data.xtype
        flipped_y = add_dim(flipped_y) |> variant_data.ytype
        push!(new_x, flipped_x)
        push!(new_y, flipped_y)
        counter += 1
        if counter % 1024 == 1
            println(counter)
        end
    end
    variant_minibatch(
        new_x,
        new_y;
        shuffle = original_shuffle,
        xtype = variant_data.xtype,
        ytype = variant_data.ytype,
    )
end

function mirror_raw_batch_item(batch_item, is_label, atype)
    arr = Array(batch_item)
    arr = reduce_dim(arr)
    flipped =
        is_label ? exchange_symmetric_joint_ids_for_flipped_labels(fliplr3D(arr)) :
        fliplr3D(arr)
    flipped = add_dim(flipped) |> atype
    return flipped
end

function mirror_data_items_for_batch(data_items)
    mirrored_data_items = map(mirror_data_item, data_items)
    return [data_items..., mirrored_data_items...]
end

function fliplr3D(img)
    copied_img = deepcopy(img)
    y_dim = size(img)[1]
    x_dim = size(img)[2]
    for y = 1:y_dim
        for x = 1:x_dim
            copied_img[y, x_dim-x+1, :] = img[y, x, :]
        end
    end
    return copied_img
end

function fliplr2D(img)
    copied_img = deepcopy(img)
    y_dim = size(img)[1]
    x_dim = size(img)[2]
    for y = 1:y_dim
        for x = 1:x_dim
            copied_img[y, x_dim-x+1] = img[y, x]
        end
    end
    return copied_img
end


function exchange_symmetric_joint_ids_for_flipped_labels(labels)
    copied_labels = deepcopy(labels)
    h, w, c = size(labels)

    for i = 1:global_num_joints
        copied_labels[:, :, i] = labels[:, :, symmetric_joint_dict[i]]
    end

    for i = global_num_joints+1:2:c
        if i < global_num_joints * 3 + 1
            j = (i - global_num_joints) / 2 |> ceil |> Int
            sym_j = symmetric_joint_dict[j]
            new_x = (sym_j - j) * 2 + i
            new_y = (sym_j - j) * 2 + 1 + i
            copied_labels[:, :, i] = labels[:, :, new_x]
            copied_labels[:, :, i+1] = labels[:, :, new_y]
        else
            j = (i - global_num_joints * 3) / 2 |> ceil |> Int
            sym_j = symmetric_joint_dict[j]
            new_x = (sym_j - j) * 2 + i
            new_y = (sym_j - j) * 2 + 1 + i
            copied_labels[:, :, i] = labels[:, :, new_x]
            copied_labels[:, :, i+1] = labels[:, :, new_y]
        end
    end
    return copied_labels
end

function reverse_all_dims(array; until_dim)
    dim_count = array |> size |> length
    for i = 1:dim_count
        if until_dim == i
            break
        end
        array = reverse(array, dims = i)
    end
    return array
end

function write_acc_results_to_csv(filename, acc_results)
    df = DataFrame(
        Ankle1 = Float32[],
        Knee1 = Float32[],
        Hip1 = Float32[],
        Hip2 = Float32[],
        Knee2 = Float32[],
        Ankle2 = Float32[],
        Wrist1 = Float32[],
        Elbow1 = Float32[],
        Shoulder1 = Float32[],
        Shoulder2 = Float32[],
        Elbow2 = Float32[],
        Wrist2 = Float32[],
        Chin = Float32[],
        TopHead = Float32[],
    )
    for i = 1:size(acc_results, 1)
        push!(df, acc_results[i, :])
    end

    CSV.write("$(filename).csv", df)

end
