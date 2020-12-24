using(LinearAlgebra)
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
    pckh = sqrt(sum(( pred .- gt ) .^2)) / head_size
    return pckh
end

# TODO: impl PCKH
# NOW it should not be that hard

# Naive Accuracy

# https://www.programmersought.com/article/7644537351/#:~:text=1.,PCK%20%2D%20Percentage%20of%20Correct%20Keypoints&text=calculates%20the%20percentage%20of%20detections,used%20as%20a%20normalized%20reference.

function naive_pck(output, y_gold; consider_threshold=output_consider_threshold, wrong_scs_cache=nothing)
    num_elements = size(y_gold)[4]
    num_joints = global_num_joints
    correct_counter = 0
    for idx in 1:num_elements
        for j in 1:num_joints
            joint_sc = output[:, :, j, idx]
            joint_sc_initial = deepcopy(joint_sc)
            joint_sc[ joint_sc .< consider_threshold] .= 0  
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
        output = Array{Float32}(sigm.(model(x)))
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
        pck = naive_pck(output, ygold; wrong_scs_cache=wrong_scs_cache)
        push!(results, pck)
    end
    final = sum(results) / length(results)
    return final / 100
end