using Knet
using Images

add_dim(x::Array) = reshape(x, (size(x)..., 1));

# Not working as expected
function concatenate_iterables(iterables...)
    return zip(iterables...) |> Iterators.flatten
end

# mutable struct Data{T}; x; y; batchsize; length; partial; imax; indices; shuffle; xsize; ysize; xtype; ytype; end
function append_to_data!(data::Knet.Train20.Data, another_data::Knet.Train20.Data)
    println(size(data.x))
data.x = cat(data.x, another_data.x, dims=2)
data.y = cat(data.y, another_data.y, dims=2)
    data.length = data.length + another_data.length
    data.imax = data.length - data.batchsize + 1
    data.indices = 1:data.length
end

function model_end_results(model, train_data, validation_data)
    train_loss = model(train_data)
    validation_loss = model(validation_data)
    train_acc = modelized_naive_pck_sigm(model, train_data)
    validation_acc = modelized_naive_pck_sigm(model, validation_data)
    end_res_dict = Dict("train_loss" => train_loss, 
        "validation_loss" => validation_loss,
        "train_acc" => train_acc,
        "validation_acc" => validation_acc
    )
    for (k, v) in end_res_dict
        println("$(k): $(v)")
    end
end

function show_scmap_on_image(image, scmap; fill_value=0, should_display=true, should_use_scmap_size=true, display_name=nothing)
    size_h = should_use_scmap_size ? size(scmap)[1] : size(image)[1]
    size_w = should_use_scmap_size ?  size(scmap)[2] : size(image)[2]
    
    output_sized_image = imresize(image, size_h, size_w);
    scmap = imresize(scmap, size_h, size_w)
    
    perm = permutedims(output_sized_image, [3, 1, 2])
    pred_joint_image = add_dim(Array(scmap));
    permuted_pred = permutedims(pred_joint_image, [3, 1, 2]);
    a1p, a2p = paddedviews(0, permuted_pred, perm);
    float_a2p = Float32.(a2p);
    float_a1p = Float32.(a1p);
    masked_a2p = deepcopy(float_a2p);
    mask = @view masked_a2p[float_a1p .> 0];
    fill!(mask, fill_value);
    colored = colorview(RGB, masked_a2p)
    if should_display
        if display_name != nothing; fetch(println(display_name)); end
        fetch(display(colored))
    end
    return colored
end

function draw_score_maps(y, image_order)
    images = []
test_y = y[:, :, :, image_order];
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
    all_joints = [
         [1, 6],
       [2, 5],
     [3, 4],
      [7, 12],
       [8, 11],
       [9, 10],
       [13],
    [14]]
 for (joint_group) in all_joints
      for j in joint_group
        permuted = permutedims(test_y[:, : , j], (1,2))
        push!(images, colorview(Gray, permuted))         
    end
end
    mosaic = mosaicview(images..., ncol=2, rowmajor=true, npad=1, fillvalue=0.3)
    display(mosaic)
    return images
end

function mosaicify(images; ncol=2, rowmajor=true, npad=1, fillvalue=0.3)
 mosaic = mosaicview(images..., ncol=ncol, rowmajor=rowmajor, npad=npad, fillvalue=fillvalue)
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