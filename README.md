# A Replication Study of "DeeperCut Part Detectors" with Knet and Julia

The aim of this project is to replicate the works of DeeperCut and then combine it with the findings of DeepLabCut. However, this goal was not realized because of unfruitful results of the replication and hence resulted with human body part detector. Human body part detector consists of deep fully convolutional neural networks and generates bottom-up proposals for body parts. Generated proposals then evaluated with both single person data in context with only single person in image and single person among other people. Finally, the pose of a single person for an input image can be drawn by using the result of the part detector. 

# Prerequisites

The implementation is in Julia 1.5.3 and Knet 1.4.6. 

- julia download link https://julialang.org/downloads/
- setting up Knet: https://denizyuret.github.io/Knet.jl/latest/install/
- using IJulia for ipynb's that are using Julia as kernel: https://github.com/JuliaLang/IJulia.jl

- list of all packages used in the project: ["ArgParse", "AutoGrad", "CSV", "CUDA", "CoordinateTransformations", "DataFrames", "FFTViews", "FileIO", "IJulia", "ImageDraw", "ImageMagick", "ImageTransformations", "ImageView", "Images", "IterTools", "JuliaFormatter", "Knet", "MAT", "NNlib", "OffsetArrays", "Plots", "Rotations", "TestImages"]

Here is how you can install a package to your Julia environment.
```bash
using Pkg
Pkg.add("Package Name")
```
# Training

## Pretrained Weights for Initial Configuration

Model uses pretrained weights from ImageNet. So those should be assigned in "modular.resnet.jl" as "model_file_path".
Here is the link for pretrained weights. https://www.vlfeat.org/matconvnet/pretrained/
Please download, you may download weight files for ResNet-50, ResNet-101, ResNet-152

## Dataset 

This project makes use of MPII Human Pose Dataset. "The dataset includes around 25K images containing over 40K people with annotated body joints. The images were systematically collected using an established taxonomy of every day human activities. Overall the dataset covers 410 human activities and each image is provided with an activity label. Each image was extracted from a YouTube video and provided with preceding and following un-annotated frames. In addition, for the test set we obtained richer annotations including body part occlusions and 3D torso and head orientations."

## Preprocessing

Downloaded dataset initially should be preprocessed by the DeeperCut's preprocessing script for rescaling and cropping.
- https://github.com/eldar/pose-tensorflow/blob/master/models/README.md#training-a-model-with-mpii-pose-dataset-single-person 

## Configuration

For configuration of training this file needs to be edited: **deeper-cut.config.jl**.
Editable parameters in the file are as follows;

```
path_to_processed_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/dataset.mat"
path_to_single_person_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-singlePerson-h400.mat"
path_to_full_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-full-h400.mat"
path_to_multi_person_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-multPerson-h400.mat"

global_scale = 0.8452830189
scale_jitter_interval = [0.85, 1.15]
pos_dist_thresh = 17

weight_decay = 0.0001
momentum_gamma = 0.9

global_locref_stdev = 7.2801
locref_loss_weight = 0.05
PCKh_range=0.5

mean_pixel = [123.68 / 255, 116.779 / 255, 103.939 / 255]
reshaped_mean_pixel = reshape(mean_pixel, (1, 1, 3, 1));
mean_pixel_255 = [123.68, 116.779, 103.939]
reshaped_mean_pixel_255 = reshape(mean_pixel_255, (1, 1, 3, 1));

use_locref_mask_weights = false
global_add_random_mirroring = false

pre_full_path = "/kuacc/users/gsoykan20/comp541_term_project/deeper-cut/results/"
```

## Training Script

**mpii.trainer.ipynb** can be used for training from the ground up. However, it might take time to load all the data in the first place. So, once you load training and test data, you may wish to save them. In order to save an object, data or model,  below code can be used. 

```
Knet.save("<data>.jld2", <data_tag>, data)
```
In order to load an object this can be used:
```
loaded_data = Knet.load("<data>.jld2", <data_tag>)
```
## Loading Pretrained Model from the Original Paper

- First you must download pretrained weights from this link: https://drive.google.com/file/d/11478RdsrcqfXRwvBqaI5aodUbbRYWn_4/view?usp=sharing

- Then you can update **get_params_from_deepercut_pretrained** function and **groups_path** variable for the path of the downloaded file in the **modular.resnet.jl**.

- Then you can run below cells of the "deepercut.weight.transfer.ipynb" ipython notebook. 

```
deepercut_with_loc_ref_101_pretrained = generate_deeper_cut(;should_use_resnet101 = true,
    should_use_resnet50 = false,
    is_loc_ref_enabled = true,
    connect_res3_to_res5 = false,
    use_deepercut_resnet101_pretrained = true);
```
```    
demo_image_path = "./google-drive/image.png"
loaded_image = load(demo_image_path);
   macfix = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(loaded_image))
    demo_img = permutedims(macfix, (2, 3, 1));
demo_img |> size
demo_img = (demo_img |> Array )  .* 255
demo_img = (demo_img |> add_dim)   .- (reshaped_mean_pixel .* 255)
# permuted_demo = permutedims(demo_img, [4, 1, 2, 3])
#demo_pred = deepercut_with_loc_ref_101_pretrained(Knet.atype()(reverse(demo_img, dims=1)));
demo_pred = deepercut_with_loc_ref_101_pretrained(Knet.atype()(demo_img));
```
```
should_use_scmap_size = false
return_single_image = true
res = show_scmap_on_image(
   ((demo_img  .+ (reshaped_mean_pixel .* 255)) |> reduce_dim) ./ 255 ,
    #((demo_img) |> reduce_dim) ./ 255 ,
    demo_pred;
    confidence_threshold=0.4,
    should_use_scmap_size= should_use_scmap_size,
    return_single_image= return_single_image,
    should_color_scmap=true,
    focus_on_argmax = true,
    add_loc_ref_offset = true,
    use_sigm = true
)
if return_single_image
    res |> display #imrotate(res, pi/2)
else    
for  r in res
  imrotate(r, π/2) |> display
end
end
    
```

# Measuring Accuracy

Measuing accuracy can be done with two methods. First one is naive PCK and second one is actual PCK.
Naive PCK measures whether argmax of the output is within the predetermined label. On the other hand, actual PCK measures distances between actual joint position and predicted joint location.

- **accuracy.jl** should be used.
- naive PCK

```
function modelized_naive_pck_sigm(model, data)
```

- PCKh

```
function modelized_PCKh_sigm(
    model,
    data,
    data_items;
    h_range = PCKh_range,
    consider_threshold = output_consider_threshold,
)
```

- **data_items** are the items read from data_set and has the annotations for head size and actual joint location annotations which is needed for PCKh measurement.

# Visualization

For the visualization, **visualizer.jl** should be used with function;

```
function visualize_part_detection_results(
    model,
    data;
    visualization_idxs = 1:10,
    pred_threshold = 0.2,
    focus_on_argmax = false,
    add_loc_ref_offset = false,
    show_only_predictions = false,
    show_body_skeleton = false,
    should_color_scmap_for_gt = true,
)
```
 
# References
```
@inproceedings{insafutdinov2016deepercut,
	author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schieke},
	title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year = {2016},
	url = {http://arxiv.org/abs/1605.03170}
    }
@inproceedings{pishchulin16cvpr,
	author = {Leonid Pishchulin and Eldar Insafutdinov and Siyu Tang and Bjoern Andres and Mykhaylo Andriluka and Peter Gehler and Bernt Schiele},
	title = {DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation},
	booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2016},
	url = {http://arxiv.org/abs/1511.06645}
}
@inproceedings{andriluka14cvpr,
               author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
               title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
               booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
               year = {2014},
               month = {June}
}

