using (LinearAlgebra)
include("mpii.annotation.reader.jl")



#=
function headSize = util_get_head_size(rect)
SC_BIAS = 0.6; % 0.8*0.75
headSize = SC_BIAS*norm([rect.x2 rect.y2] - [rect.x1 rect.y1]);
end
=#

function get_head_size(corner_one, corner_two)
    # TODO: This can be parameterized
    SC_BIAS = 0.6
    head_size = SC_BIAS * norm(corner_one .- corner_two)
    return head_size
end

#=
function dist = getDistPCKh(pred,gt,refDist)

assert(size(pred,1) == size(gt,1) && size(pred,2) == size(gt,2) && size(pred,3) == size(gt,3));
assert(size(refDist,1) == size(gt,3));

dist = nan(1,size(pred,2),size(pred,3));

for imgidx = 1:size(pred,3)
    
    % distance to gt joints
    dist(1,:,imgidx) = sqrt(sum((pred(:,:,imgidx) - gt(:,:,imgidx)).^2,1))./refDist(imgidx);

end
=#

function get_distance_in_PCKh(pred, gt, head_size)
    pckh = sqrt(sum((pred .- gt) .^ 2)) / head_size
    return pckh
end

# TODO: impl PCKH
# NOW it should not be that hard

# Data Items needs to be scaled beforehand
function create_acc_dict()
    person_typed_h_ranged_joint_by_joint_accuracy_dict = Dict()
    person_types = ["single", "multi", "mixed"]
    h_ranges = [hr for hr = 0:0.1:1]
    person_typed_h_ranged_joint_by_joint_accuracy_dict = Dict()

    for person_type in person_types
        h_ranged_joint_by_joint_accuracy_dict = Dict()
        for h in h_ranges
            joint_by_joint_accuracy_dict = Dict()
            # (accuracate_preds / total_number_of_gt)
            for j = 1:global_num_joints
                joint_by_joint_accuracy_dict[j] = (0, 0)
            end
            h_ranged_joint_by_joint_accuracy_dict[h] = joint_by_joint_accuracy_dict
        end
        person_typed_h_ranged_joint_by_joint_accuracy_dict[person_type] =
            h_ranged_joint_by_joint_accuracy_dict
    end
    return person_typed_h_ranged_joint_by_joint_accuracy_dict
end

function original_pckh(
    output,
    data_items;
    h_range = PCKh_range,
    consider_threshold = output_consider_threshold,
    offmat = nothing,
    person_typed_h_ranged_joint_by_joint_accuracy_dict = Dict(),
)

    num_elements = size(output)[4]
    num_joints = global_num_joints
    correct_counter = 0
    non_existant_joint_counter = 0

    for idx = 1:num_elements
        data_item = data_items[idx]
        visible_joint_ids = data_item.joints[:, 1]

        head_corner_one = (data_item.annorect["x1"], data_item.annorect["y1"])
        head_corner_two = (data_item.annorect["x2"], data_item.annorect["y2"])
        head_size = get_head_size(head_corner_one, head_corner_two)

        for j = 1:num_joints
            joint_sc = output[:, :, j, idx]
            joint_sc[joint_sc.<consider_threshold] .= 0
            max_coord = argmax(joint_sc)

            if !(j in visible_joint_ids)
                non_existant_joint_counter += 1
            else
                data_joint_idx = findlast(x -> x[1] == j, data_item.joints)[1]
                joint_gold = data_item.joints[data_joint_idx, :][2:3]

                pred = (max_coord[2], max_coord[1])

                pred_f8 = pred .* preprocess_stride .+ 0.5 * preprocess_stride

                # TODO: Let's check if coords are on same cartesian

                if offmat != nothing
                    dx = offmat[:, :, 2*j-1, idx][max_coord] * global_locref_stdev
                    dy = offmat[:, :, 2*j, idx][max_coord] * global_locref_stdev
                    pred_f8 = (pred_f8[1] + dx, pred_f8[2] + dy)
                end

                dist_PCKh = get_distance_in_PCKh(pred_f8, joint_gold, head_size)

                if dist_PCKh <= h_range
                    correct_counter += 1
                end

                person_types = ["single", "multi", "mixed"]
                h_ranges = [hr for hr = 0:0.1:1]
                for person_type in person_types
                    if (data_item.is_single && person_type == "single") ||
                       (!data_item.is_single && person_type == "multi") ||
                       (person_type == "mixed")

                        h_ranged_joint_by_joint_accuracy_dict =
                            person_typed_h_ranged_joint_by_joint_accuracy_dict[person_type]
                        for h in h_ranges
                            current_values_correct =
                                h_ranged_joint_by_joint_accuracy_dict[h][j][1]
                            current_values_total =
                                h_ranged_joint_by_joint_accuracy_dict[h][j][2]
                            current_values_total += 1
                            if dist_PCKh <= h
                                current_values_correct += 1
                            end
                            h_ranged_joint_by_joint_accuracy_dict[h][j] =
                                (current_values_correct, current_values_total)
                        end
                    end
                end
            end
        end
    end

    acc =
        100.0 * correct_counter / ((num_elements * num_joints) - non_existant_joint_counter)
    return (acc, person_typed_h_ranged_joint_by_joint_accuracy_dict)
end

function modelized_PCKh_sigm(
    model,
    data,
    data_items;
    h_range = PCKh_range,
    consider_threshold = output_consider_threshold,
)
    results = []
    acc_dist_dict = create_acc_dict()

    original_x_data = Knet.atype()(reshape(data.x, data.xsize))
    batchsize = data.batchsize
    total_element_count = size(original_x_data)[4]

    batch_coefficient = floor(total_element_count / batchsize) |> Int
    total_element_count = batch_coefficient * batchsize

    for i = 1:batchsize:total_element_count
        datapart = original_x_data[:, :, :, i:i+batchsize-1]
        dataitem_part = data_items[i:i+batchsize-1]

        output = model(datapart)
        part_detection_scores = output[:, :, 1:global_num_joints, :]
        part_detection_probs = Array{Float32}(sigm.(part_detection_scores))

        channel_dim = size(output)[end-1]
        offmat = nothing
        if channel_dim >= 3 * global_num_joints
            offmat = output[:, :, global_num_joints+1:global_num_joints*3, :]
        end

        pck, acc_dist_dict = original_pckh(
            part_detection_probs,
            dataitem_part;
            h_range = h_range,
            consider_threshold = consider_threshold,
            offmat = offmat,
            person_typed_h_ranged_joint_by_joint_accuracy_dict = acc_dist_dict,
        )

        push!(results, pck)
    end

    final = sum(results) / length(results)
    final /= 100
    return (final, acc_dist_dict)
end

# Naive Accuracy

# https://www.programmersought.com/article/7644537351/#:~:text=1.,PCK%20%2D%20Percentage%20of%20Correct%20Keypoints&text=calculates%20the%20percentage%20of%20detections,used%20as%20a%20normalized%20reference.

function naive_pck(
    output,
    y_gold;
    consider_threshold = output_consider_threshold,
    wrong_scs_cache = nothing,
)
    num_elements = size(y_gold)[4]
    num_joints = global_num_joints
    correct_counter = 0
    for idx = 1:num_elements
        for j = 1:num_joints
            joint_sc = output[:, :, j, idx]
            joint_sc_initial = deepcopy(joint_sc)
            joint_sc[joint_sc.<consider_threshold] .= 0
            max_coord = argmax(joint_sc)
            gt_sc = y_gold[:, :, j, idx]
            if gt_sc[max_coord] == 1
                correct_counter += 1
            elseif joint_sc[max_coord] == 0 && maximum(gt_sc) == 0
                correct_counter += 1
            else
                if wrong_scs_cache != nothing
                    push!(wrong_scs_cache[1], gt_sc)
                    push!(wrong_scs_cache[2], joint_sc)
                    push!(wrong_scs_cache[3], joint_sc_initial)
                end
            end

        end
    end
    acc = 100.0 * correct_counter / (num_elements * num_joints)
    return acc
end

function modelized_naive_pck_sigm(model, data)
    results = []
    for (x, ygold) in data
        ygold = Array{Float32}(ygold)
        ygold = ygold[:, :, 1:global_num_joints, :]

        output = model(x)
        output = output[:, :, 1:global_num_joints, :]

        output = Array{Float32}(sigm.(output))
        pck = naive_pck(output, ygold)
        push!(results, pck)
    end
    final = sum(results) / length(results)
    return final / 100
end

function modelized_naive_pck_sigm_with_wrong_cache(model, data, wrong_scs_cache)
    results = []
    for (x, ygold) in data
        ygold = Array{Float32}(ygold)
        output = Array{Float32}(sigm.(model(x)))
        pck = naive_pck(output, ygold; wrong_scs_cache = wrong_scs_cache)
        push!(results, pck)
    end
    final = sum(results) / length(results)
    return final / 100
end
