using Knet
using Images

add_dim(x::Array) = reshape(x, (size(x)..., 1));

reduce_dim(x::Array) = reshape(x, (size(x)[1:end-1]));

# Not working as expected
function concatenate_iterables(iterables...)
    return zip(iterables...) |> Iterators.flatten
end

# mutable struct Data{T}; x; y; batchsize; length; partial; imax; indices; shuffle; xsize; ysize; xtype; ytype; end
function append_to_data!(data::Knet.Train20.Data, another_data::Knet.Train20.Data)
    println(size(data.x))
    data.x = cat(data.x, another_data.x, dims = 2)
    data.y = cat(data.y, another_data.y, dims = 2)
    data.length = data.length + another_data.length
    data.imax = data.length - data.batchsize + 1
    data.indices = 1:data.length
end

function model_end_results(
    model,
    train_data,
    validation_data;
    train_data_items,
    validation_data_items,
)
    train_loss = model(train_data)
    validation_loss = model(validation_data)
    train_acc = modelized_naive_pck_sigm(model, train_data)
    validation_acc = modelized_naive_pck_sigm(model, validation_data)

    train_acc_pckh = nothing
    validation_acc_pckh = nothing

    if train_data_items != nothing && validation_data_items != nothing
        train_acc_pckh = modelized_PCKh_sigm(model, train_data, train_data_items)[1]
        validation_acc_pckh =
            modelized_PCKh_sigm(model, validation_data, validation_data_items)[1]
    end

    end_res_dict = Dict(
        "train_loss" => train_loss,
        "validation_loss" => validation_loss,
        "train_acc" => train_acc,
        "validation_acc" => validation_acc,
        "train_acc_PCKh" => train_acc_pckh,
        "validation_PCKh" => validation_acc_pckh,
    )
    for (k, v) in end_res_dict
        println("$(k): $(v)")
    end
end

function display_deepercut_input_image(input_image)
    colorviews = []

    function single_display(single_img)
        perm = permutedims(single_img, [3, 1, 2])
        push!(colorviews, colorview(RGB, perm))
    end

    input_image_sizes = size(input_image)
    if input_image_sizes |> length == 4
        for i = 1:input_image_sizes[end]
            img = input_image[:, :, :, i]
            single_display(img)
        end
    else
        single_display(input_image)
    end

    return colorviews
end

#=
# TODO: This can be used for visualization purposes 
arrange_img_damacfix = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(loaded_img));
converted_img = convert(Array{Float32}, arrange_img_damacfix);
img_v = @view converted_img[:, 38:75 , 112:177] # blue;
fill!(img_v, 0) # b;
=#

function show_scmap_on_image(
    image,
    scmap;
    should_display = false,
    should_use_scmap_size = true,
    display_name = nothing,
        confidence_threshold = 0,
        focus_on_arg_max = false,
        add_loc_ref_offset = false
)
    scmap = sigm.(scmap)
    colored_images = []
    size_h = should_use_scmap_size ? size(scmap)[1] : size(image)[1]
    size_w = should_use_scmap_size ? size(scmap)[2] : size(image)[2]
    output_sized_image = imresize(image, size_h, size_w)
        
      all_joints = [[1, 6], [2, 5], [3, 4], [7, 12], [8, 11], [9, 10], [13], [14]]
        for (joint_group) in all_joints
            for joint_id in joint_group
        
        scmap_slice = scmap[:,:, joint_id]
        scmap_slice = imresize(scmap_slice, size_h, size_w)
        perm = permutedims(output_sized_image, [3, 1, 2])
                
        if focus_on_arg_max != true
            color_idx = findall(scmap_slice .> confidence_threshold)
        elseif add_loc_ref_offset
        
        else
            color_idx = argmax(scmap_slice)
        end
                
        
        
            
            
        img_r = @view perm[1, color_idx]
        img_g = @view perm[2, color_idx]
        img_b = @view perm[3, color_idx] 
        
        fill!(img_r, (joint_id + 4) / (global_num_joints + 4) * 1.2)
        fill!(img_b, 1.4 - ( (joint_id + 4) / (global_num_joints + 4) * 1.4) )
        fill!(img_g, (joint_id + 4) / (global_num_joints + 4) * 1)
        
        colored = colorview(RGB, perm)
        push!(colored_images, colored)
    end
    end
    
   return colored_images
end

function draw_score_maps(y)
    images = []

    function draw_score_maps_for_single(y_single)
        #= all_joints = Dict(
            "ankle" => [1, 6],
            "knee" => [2, 5],
            "hip" => [3, 4],
           "wrist" => [7, 12],
            "elbow" => [8, 11],
            "shoulder" => [9, 10],
            "chin" => [13],
         "forehead" => [14]
        ) =#
        images_for_single = []
        all_joints = [[1, 6], [2, 5], [3, 4], [7, 12], [8, 11], [9, 10], [13], [14]]
        for (joint_group) in all_joints
            for j in joint_group
                permuted = permutedims(y_single[:, :, j], (1, 2))
                push!(images_for_single, colorview(Gray, permuted))
            end
        end
        mosaic = mosaicview(
            images_for_single...,
            ncol = 2,
            rowmajor = true,
            npad = 1,
            fillvalue = 0.3,
        )
        display(mosaic)
        push!(images, images_for_single...)
    end

    y_sizes = size(y)
    if y_sizes |> length == 4
        for i = 1:y_sizes[end]
            single_y = y[:, :, :, i]
            draw_score_maps_for_single(single_y)
        end
    else
        draw_score_maps_for_single(y)
    end

    return images
end

function mosaicify(images; ncol = 2, rowmajor = true, npad = 1, fillvalue = 0.3)
    mosaic = mosaicview(
        images...,
        ncol = ncol,
        rowmajor = rowmajor,
        npad = npad,
        fillvalue = fillvalue,
    )
    display(mosaic)
end

function get_object_tag(obj)
    obj_field_names = fieldnames(typeof(obj))
    has_tag = :tag in obj_field_names
    if has_tag
        return obj.tag
    else
        return nothing
    end
end

function read_accuracy_results(acc_dist_dict)
    person_types = ["single", "multi", "mixed"]
    h_ranges = [hr for hr = 0:0.1:1]
    joint_ids = [j for j = 1:global_num_joints]
    acc_res = zeros(3 * 11, global_num_joints)
    for (h_i, h_range) in enumerate(h_ranges)
        for joint_id in joint_ids
            for (p_i, person_type) in enumerate(person_types)
                accurate = acc_dist_dict[person_type][h_range][joint_id][1]
                total = acc_dist_dict[person_type][h_range][joint_id][2]
                perc = 100.0 * accurate / total
                acc_res[(h_i-1)*3+p_i, joint_id] = round(perc, digits = 2)
            end
        end
    end
    return acc_res
end

function write_results_to_file(filename, results_kv_tuples...)
    open(filename, "a") do io
        for (k, v) in results_kv_tuples
            write(io, "$(k): $(v)")
            write(io, "\n")
        end
    end
end
