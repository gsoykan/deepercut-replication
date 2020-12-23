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
path_to_processed_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/dataset.mat"
global_num_joints = 14

# Total image count => 28883
validation_image_count = 8
train_image_count = 256
read_image_w = 256
read_image_h = 256
max_image_number_to_read = validation_image_count + train_image_count

global_scale = 0.8452830189
preprocess_stride = 8
pos_dist_thresh = 17
output_consider_threshold = 0.25
# TODO: learn how this was computed 
global_locref_stdev = 7.2801

function read_cropped_mpii_annotations(;should_shuffle=false)
    file = matopen(path_to_processed_mat)
    dataset = read(file, "dataset")
    close(file)
    reformat_dataset!(dataset, should_shuffle)
    return dataset
end

function reformat_dataset!(dataset, should_shuffle) 
    zipped = zip(dataset["image"], dataset["size"], dataset["joints"])
    collected = collect(zipped)
    if should_shuffle
        shuffle!(collected)
    end
    vecced = vec(collected)
    dataset["size"] = map(e -> e[2],vecced)
    dataset["image"] = map(e -> e[1],vecced)
    dataset["joints"] = map(e -> e[3],vecced)
    return dataset
end

function get_from_dataset(dataset, initial_index=1, last_index=max_image_number_to_read)   
    sizes = dataset["size"][initial_index: last_index]
    images = dataset["image"][initial_index: last_index]
    joints = dataset["joints"][initial_index: last_index]
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