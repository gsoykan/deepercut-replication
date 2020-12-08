include("mpii.annotation.reader.jl")

function naive_pck(output, y_gold)
    num_elements = size(y_gold)[4]
    num_joints = global_num_joints
    correct_counter = 0
    for idx in 1:num_elements
        for j in 1:num_joints
            joint_sc = output[:, :, j, idx]
            gt_sc = y_gold[:, :, j, idx]
            joint_sc[ joint_sc .< 0.5] .= 0  
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