include("mpii.annotation.reader.jl")
include("../helper.jl")
include("../models.jl")
include("loss.jl")
include("../utils.jl")
using Images, FileIO, ImageDraw
using Knet

function run_visualizer(
    model_location_name = "29-12-20-training_model",
    Knet_data_location_name = "29-12-20-dval";
    visualization_idxs = 1:10,
    pred_threshold = 0.2,
    focus_on_argmax = false,
    add_loc_ref_offset = false,
    show_only_predictions = false,
    show_body_skeleton = false,
)
    model = Knet.load("$(model_name).jld2", model_name)
    dval = Knet.load("$(Knet_data_location_name).jld2", Knet_data_location_name)

    visualize_part_detection_results(
        model,
        data;
        pred_threshold = pred_threshold,
        focus_on_argmax = focus_on_argmax,
        add_loc_ref_offset = add_loc_ref_offset,
        show_only_predictions = show_only_predictions,
        visualization_idxs = visualization_idxs,
        show_body_skeleton = show_body_skeleton,
    )
end

# TODO: Can we add heatmap like visualization
function visualize_part_detection_results(
    model,
    data;
    visualization_idxs = 1:10,
    pred_threshold = 0.2,
    focus_on_argmax = false,
    add_loc_ref_offset = false,
    show_only_predictions = false,
    show_body_skeleton = false,
)
    data_x = Knet.atype()(reshape(data.x[:, visualization_idxs], data.xsize))
    data_y = Knet.atype()(reshape(data.y[:, visualization_idxs], data.ysize))
    output = model(data_x)
    output = Array(output)
    for idx in 1:length(visualization_idxs)        
        img = Array(data_x[:, :, :, idx])
        if !show_only_predictions
            gt_scmap = Array(data_y[:, :, :, idx])
            gt_images = show_scmap_on_image(
                img,
                gt_scmap;
                confidence_threshold = 0.5,
                should_use_scmap_size = false,
                return_single_image = show_body_skeleton,
            )

            if (show_body_skeleton)
              gt_images =  draw_body_skeleton(gt_images, gt_scmap)
            end

            println("****** GROUND TRUTH ******")
           show_body_skeleton ? display(gt_images) : mosaicify(gt_images)
        end
        pred_scmap = output[:, :, :, idx]
        pred_images = show_scmap_on_image(
            img,
            pred_scmap;
            confidence_threshold = pred_threshold,
            should_use_scmap_size = false,
            focus_on_argmax = focus_on_argmax,
            add_loc_ref_offset = add_loc_ref_offset,
            return_single_image = show_body_skeleton,
        )

        if (show_body_skeleton)
           pred_images = draw_body_skeleton(pred_images, pred_scmap)
        end

        println("****** PART PREDICTIONS ******")
        show_body_skeleton ? display(pred_images) : mosaicify(pred_images)
    end
end

function draw_body_skeleton(img, scmap)
    colors = [
        RGB{Float32}(0.9, 0.1, 0.1),
        RGB{Float32}(0.1, 0.9, 0.1),
        RGB{Float32}(0.1, 0.1, 0.9),
        RGB{Float32}(0.4, 0.1, 0.1),
        RGB{Float32}(0.1, 0.4, 0.1),
        RGB{Float32}(0.1, 0.1, 0.4),
        RGB{Float32}(0.2, 0.6, 0.2),
    ]

    all_joints = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [13, 14],
    ]

    for (i, (joint_group)) in enumerate(all_joints)
        color_idx = i % length(colors) + 1
        color = colors[color_idx]
        joint_1 = joint_group[1]
        joint_2 = joint_group[2]
        pred_f8_1 = fetch_loc_reffed_max_point(scmap, joint_1)
        pred_f8_2 = fetch_loc_reffed_max_point(scmap, joint_2)
        img = draw!(img, LineSegment(pred_f8_1, pred_f8_2), color)
    end
    return img
end
