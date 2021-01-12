# cfg.all_joints = [      [0, 5], [1, 4], [2, 3], [6, 11], [7, 10],  [8, 9],     [12],      [13]]
# cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
# cfg.num_joints = 14

# TODO: add this to config
path_to_processed_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/dataset.mat"
path_to_single_person_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-singlePerson-h400.mat"
path_to_full_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-full-h400.mat"
path_to_multi_person_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-multPerson-h400.mat"
global_num_joints = 14
# all_joints = [[1, 6], [2, 5], [3, 4], [7, 12], [8, 11], [9, 10], [13], [14]]
symmetric_joint_dict = Dict(
    1 => 6,
    6 => 1,
    2 => 5,
    5 => 2,
    3 => 4,
    4 => 3,
    7 => 12,
    12 => 7,
    8 => 11,
    11 => 8,
    9 => 10,
    10 => 9,
    13 => 13,
    14 => 14 )

# Total image count => 28883
# multi-person image count => 9698
# single-person image count => 19185 ( 17440 - 1744 )
test_image_count = 1000
validation_image_count = 64 #28883 - 25600 - 1000
train_image_count =  1024 # 25600
read_image_w = 512
read_image_h = 512
max_image_number_to_read = validation_image_count + train_image_count + test_image_count

global_scale = 0.8452830189
preprocess_stride = 8
pos_dist_thresh = 17

# TODO: output consider threshold might not be needed from now on
output_consider_threshold = 0.0
# TODO: learn how this was computed 
global_locref_stdev = 7.2801
locref_loss_weight = 0.05
PCKh_range=0.5
whole_dataset_count = 28883
single_dataset_count = 19185
mean_pixel = [123.68 / 255, 116.779 / 255, 103.939 / 255]
reshaped_mean_pixel = reshape(mean_pixel, (1, 1, 3, 1));
use_locref_mask_weights = false
global_add_random_mirroring = true

pre_full_path = "/kuacc/users/gsoykan20/comp541_term_project/deeper-cut/results/"