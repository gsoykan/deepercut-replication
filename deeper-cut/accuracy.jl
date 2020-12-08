include("mpii.annotation.reader.jl")

# https://www.programmersought.com/article/7644537351/#:~:text=1.,PCK%20%2D%20Percentage%20of%20Correct%20Keypoints&text=calculates%20the%20percentage%20of%20detections,used%20as%20a%20normalized%20reference.

function naive_pck(output, y_gold; consider_threshold=0.5)
    num_elements = size(y_gold)[4]
    num_joints = global_num_joints
    correct_counter = 0
    for idx in 1:num_elements
        for j in 1:num_joints
            joint_sc = output[:, :, j, idx]
            gt_sc = y_gold[:, :, j, idx]
            joint_sc[ joint_sc .< consider_threshold] .= 0  
            max_coord = argmax(joint_sc)
            if gt_sc[max_coord] == 1
                correct_counter += 1
            elseif joint_sc[max_coord] == 0 && maximum(gt_sc) == 0
                correct_counter += 1
            end
        end
    end
    return 100.0 * correct_counter / (num_elements * num_joints)
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
    return   final / 100
end