# A Replication Study of "DeeperCut Part Detectors" with Knet and Julia Lang

The aim of this project is to replicate the works of DeeperCut and then combine it with the findings of DeepLabCut. However, this goal was not realized because of unfruitful results of the replication and hence resulted with human body part detector. Human body part detector consists of deep fully convolutional neural networks and generates bottom-up proposals for body parts. Generated proposals then evaluated with both single person data in context with only single person in image and single person among other people. Finally, the pose of a single person for an input image can be drawn by using the result of the part detector. 

## Prerequisites

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
TODO: ADD required files

## References
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
