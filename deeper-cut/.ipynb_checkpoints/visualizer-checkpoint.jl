include("mpii.annotation.reader.jl")
include("../helper.jl")
include("../models.jl")
include("loss.jl")
include("../utils.jl")
using Images, FileIO
using Knet


function visualize_part_detection_results(
    model,
    data;
    visualization_idxs = 1:10,
    pred_threshold = 0.2,
)
    data_x = Knet.atype()(reshape(data.x[:, visualization_idxs], data.xsize))
    data_y = Knet.atype()(reshape(data.y[:, visualization_idxs], data.ysize))
    output = model(data_x)
    output = Array(output)
    for idx in visualization_idxs
        img = Array(dval_29_x[:, :, :, idx])
        gt_images = show_scmap_on_image(
            img,
            Array(data_y[:, :, :, idx]);
            confidence_threshold = 0.5,
            should_use_scmap_size = false,
        )
        pred_images = show_scmap_on_image(
            img,
            output[:, :, :, idx];
            confidence_threshold = pred_threshold,
            should_use_scmap_size = false,
        )
        println("****** GROUND TRUTH ******")
        mosaicify(gt_images)
        println("****** PART PREDICTIONS ******")
        mosaicify(pred_images)
    end
end
