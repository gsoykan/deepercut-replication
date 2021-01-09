using Distributed
using Images, FileIO
include("mpii.annotation.reader.jl")

# TODO: prepare an image 
# then stack them to create whole dataset 
# ısValid size check atılabilir

# struct DataItem 
 #   id::Int
  #  path::String
   # size::Array{Float64}
    # joints::Array
# end

# If w has dimensions (W1,W2,...,Cx,Cy)
# and x has dimensions (X1,X2,...,Cx,N), the result y will have dimensions (Y1,Y2,...,Cy,N)
# where Cx is the number of input channels, Cy is the number of output channels,
# N is the number of instances, and Wi,Xi,Yi are spatial dimensions with Yi determined by:

function make_even(x)
    return isodd(x) ?  x + 1 : x
end

function preprocess_dataset(dataset; use_global_scaling = false)
    #max_h = findmax(map(data -> data.size[2], dataset))[1]
    #max_w = findmax(map(data -> data.size[3], dataset))[1]
    raw_preprocessed = map(element -> preprocess_single_image_features(element[2]; use_global_scaling = use_global_scaling, counter = element[1]), enumerate(dataset))
    return raw_preprocessed
end

# TODO: update global scaling for custom iterator 
function preprocess_single_image_features(img_data; use_global_scaling = false, counter)
    
    if counter % 1024 == 1
        println("preprocessing $(counter)")
    end
    
    img_path = img_data.path
    loaded_image = load(img_path)
    
    if use_global_scaling
        scale_h = global_scale
        scale_w = global_scale
        scales = (scale_h, scale_w)
        resized_image = imresize(loaded_image, ratio = global_scale)
    else
        scaled_max_w = read_image_w
        scaled_max_h = read_image_h
        scale_h = convert(Float32, scaled_max_h / img_data.size[3]) 
        scale_w = convert(Float32,  scaled_max_w / img_data.size[2]) 
        scales = (scale_h, scale_w)
        resized_image = imresize(loaded_image, scaled_max_h, scaled_max_w)
    end   

    arranged_img_data = arrange_img_data(resized_image)
    scaled_img_size = size(arranged_img_data)[1:2]
    sm_size = Int.(ceil.(scaled_img_size ./ (preprocess_stride * 2))) .* 2  
   
    temp_holder = img_data.joints[:, 2:3]
    temp_holder = reshape(temp_holder, (:, 2))
    temp = zeros(size(img_data.joints))
    for i in 1:size(temp_holder)[1]
        j1 =  temp_holder[i, 1] * scales[1]
        j2 =  temp_holder[i, 2] * scales[2]
        temp[i, 2] =  j1
        temp[i, 3] = j2
    end

    scaled_joints = temp[:, 2:3]
    
    joint_ids = img_data.joints[:, 1]

    (scmap, scmap_weights, locref_map, locref_mask) = compute_targets_weights(joint_ids, scaled_joints, sm_size)
    
    # TODO: check if this is working or not
    img_data = DataItem(img_data, scaled_img_size[1], scaled_img_size[2])
    
    return (img_data, arranged_img_data, scmap, scmap_weights, locref_map, locref_mask)
end

function arrange_img_data(img)
    b1 = img     
        # ad-hoc solution for Mac-OS image
    macfix = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(b1))
    c1 = permutedims(macfix, (3, 2, 1))
    w, h, c = size(c1);
    d1 = convert(Array{Float32}, c1)
    f1 = reshape(d1[:,:,1:3], (w, h, 3, 1))
        # H - W - C - N
    g1 = permutedims(f1, [2,1,3,4])
    g1
end

# TODO: Check pose_dataset.py for pairwise implementation - currently skipped 
function compute_targets_weights(joint_ids, coords, size)
    stride = preprocess_stride
    dist_thresh = pos_dist_thresh
    num_joints = global_num_joints
    half_stride = stride / 2
    scmap = zeros(size..., num_joints)
    
    locref_shape = (size..., num_joints * 2)
    locref_mask = zeros(locref_shape)
    locref_map = zeros(locref_shape)
    
    dist_thresh_sq = abs2(dist_thresh)
    
    width = size[2]
    height = size[1]
    
    for (k, j_id) in enumerate(joint_ids)
        joint_pt = coords[k, :]
        j_x  = joint_pt[1]
        j_y = joint_pt[2]
        
        # don't loop over entire heatmap, but just relevant locations
        
        j_x_sm = round((j_x - half_stride) / stride) |> Int
        j_y_sm = round((j_y - half_stride) / stride) |> Int
        min_x = round(max(j_x_sm - dist_thresh - 1, 1)) |> Int
        max_x = round(min(j_x_sm + dist_thresh + 1, width)) |> Int
        min_y = round(max(j_y_sm - dist_thresh - 1, 1)) |> Int
        max_y = round(min(j_y_sm + dist_thresh + 1, height)) |> Int
        
        for j in min_y:max_y
            pt_y = j * stride + half_stride
            for i in min_x:max_x
                pt_x = i * stride + half_stride
                dx = j_x  - pt_x
                dy = j_y - pt_y
                dist = abs2(dx) + abs2(dy)
                
                if dist <= dist_thresh_sq
                    locref_scale = 1.0 / global_locref_stdev
                    current_normalized_dist = dist * abs2(locref_scale)
                    
                    prev_normalized_dist = locref_map[j, i, j_id * 2 + 1]^2 + locref_map[j, i, j_id * 2 + 2]^2 
                    
                    update_scores = (scmap[j, i, j_id + 1] == 0) || prev_normalized_dist > current_normalized_dist            
                    
                    if update_scores
                        set_locref(locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy)
                    end
                    scmap[j, i, j_id + 1] = 1
                end
            end
        end
    end
    scmap_weights = compute_scmap_weights(size, joint_ids)
    return (scmap, scmap_weights, locref_map, locref_mask)    
end    

function set_locref(locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy)
    locref_mask[j, i, j_id * 2 + 1] = 1
    locref_mask[j, i, j_id * 2 + 2] = 1
    locref_map[j, i, j_id * 2 + 1] = dx * locref_scale
    locref_map[j, i, j_id * 2 + 2] = dy * locref_scale
end

function compute_scmap_weights(scmap_size, joint_ids)
    # TODO: This can be extra feature => Check pose_dataset.py 402 from pose-tensorflow
#     if cfg.weigh_only_present_joints:
#             weights = np.zeros(scmap_shape)
#             for person_joint_id in joint_id:
#                 for j_id in person_joint_id:
#                     weights[:, :, j_id] = 1.0
#         else:
    weights = ones(scmap_size)
    return weights
end
