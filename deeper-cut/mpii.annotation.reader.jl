using MAT
using Distributed

struct DataItem 
    id::Int
    path::String
    size::Array{Float64}
    joints::Array
end

# cfg.all_joints = [      [0, 5], [1, 4], [2, 3], [6, 11], [7, 10],  [8, 9],     [12],      [13]]
# cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
# cfg.num_joints = 14

# TODO: add this to config
path_to_processed_mat = "/home/gsoykan/Desktop/comp541/mpii-dataset/cropped/dataset.mat"
global_num_joints = 14

validation_image_count = 32
train_image_count = 256
max_image_number_to_read = validation_image_count + train_image_count

global_scale = 0.8452830189
preprocess_stride = 32
pos_dist_thresh = 17
# TODO: learn how this was computed 
global_locref_stdev = 7.2801

function read_cropped_mpii_annotations()
    file = matopen(path_to_processed_mat)
    dataset = read(file, "dataset")
    close(file)
    return dataset
end

function get_dataset()
    dataset = read_cropped_mpii_annotations()
    sizes = dataset["size"][:, 1:max_image_number_to_read]
    images = dataset["image"][:, 1:max_image_number_to_read]
    joints = dataset["joints"][:, 1:max_image_number_to_read]
    data_items = pmap(raw_data_to_data_item, enumerate(zip(images, sizes, joints)))
    return data_items
end

function raw_data_to_data_item(indexed_raw_data) 
    i = indexed_raw_data[1]
    img_path = indexed_raw_data[2][1]
    img_size = indexed_raw_data[2][2]
    joints_data = indexed_raw_data[2][3][:][1]
    joints_ids = joints_data[:, 1]
    id_check = all(id -> id < global_num_joints, joints_ids)
    if (!id_check)
        throw(DomainError(joints_data, "invalid joint id"))
    end
    return DataItem(i, img_path, img_size, joints_data)
end