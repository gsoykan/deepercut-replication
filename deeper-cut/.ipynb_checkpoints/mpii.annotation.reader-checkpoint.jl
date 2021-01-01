using MAT
using Distributed
using Random

struct DataItem 
    id::Int
    path::String
    size::Array{Float64}
    joints::Array
    is_single::Bool
    annorect::Dict{String, Any}
    
    function DataItem(
        id::Int,
    path::String,
    size::Array{Float64},
    joints::Array,
    is_single::Bool,
    annorect::Dict{String, Any}
        )
        return new(id, path, size, joints, is_single, annorect)
    end
    
    function DataItem(original_data_item::DataItem, new_h_size, new_w_size; shift_idx=true)
        copied_data_item = deepcopy(original_data_item)
        if (shift_idx)
            copied_data_item.joints[:, 1] .+= 1
        end
        
         scale_h = convert(Float32, new_h_size / copied_data_item.size[2]) 
    scale_w = convert(Float32,  new_w_size / copied_data_item.size[3]) 
        
        copied_data_item.size[2] = new_h_size
        copied_data_item.size[3] = new_w_size
        
        copied_data_item.joints[:, 2] = round.(Array{Float32}(copied_data_item.joints[:, 2]) .*= scale_w)
         copied_data_item.joints[:, 3] = round.(Array{Float32}(copied_data_item.joints[:, 3]) .*= scale_h)
        
        # Changing annorect is enough, no need to scale annopoints 
        copied_data_item.annorect["x1"] *= scale_w
        copied_data_item.annorect["x2"] *= scale_w
        copied_data_item.annorect["y1"] *= scale_h
        copied_data_item.annorect["y2"] *= scale_h
        
       return copied_data_item 
    end
end



# cfg.all_joints = [      [0, 5], [1, 4], [2, 3], [6, 11], [7, 10],  [8, 9],     [12],      [13]]
# cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
# cfg.num_joints = 14

# TODO: add this to config
path_to_processed_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/dataset.mat"
path_to_single_person_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-singlePerson-h400.mat"
path_to_full_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-full-h400.mat"
path_to_multi_person_mat = "/userfiles/gsoykan20/mpii_human_pose/cropped/annolist-multPerson-h400.mat"
global_num_joints = 14

# Total image count => 28883
# multi-person image count => 9698
# single-person image count => 19185 ( 17440 - 1744 )
test_image_count = 1000
validation_image_count = 64 #28883 - 25600 - 1000
train_image_count =  1024 # 25600
read_image_w = 128
read_image_h = 128
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
    
    image_metas = find_annorect_and_is_single_info(dataset["image"]);
    dataset["is_single"] = image_metas[:, 1]
    dataset["annorect"] = image_metas[:, 2]
    return dataset
end

function find_annorect_and_is_single_info(dataset_images)
    file = matopen(path_to_multi_person_mat);
    multi_person = read(file);
    close(file);
    
    file = matopen(path_to_single_person_mat);
    single_person = read(file);
    close(file);
    
    single_image_paths = map(x -> x["name"], single_person["annolist"]["image"])
    single_image_paths = vec(single_image_paths);
    
    multi_image_paths = map(x -> x["name"], multi_person["annolist"]["image"])
    multi_image_paths = vec(multi_image_paths);
    
    image_metas = map(image_path -> find_image_idx_in_single_multi_meta(image_path, single_image_paths, multi_image_paths, single_person, multi_person) ,dataset_images)
    
    vcatted_image_metas = vcat(image_metas...);
    return vcatted_image_metas
end

function find_image_idx_in_single_multi_meta(image_name, single_paths, multi_paths, single_person, multi_person)
    is_single = true;
    found_idx = findlast(x -> x == image_name ,single_paths)
    if found_idx == nothing 
        found_idx = findlast(x -> x == image_name, multi_paths)
        is_single = false
    elseif found_idx == nothing
        throw(DomainError(image_name, "image path not found in netiher single nor multi"))
    end
    
    if is_single
        annorect = single_person["annolist"]["annorect"][1, found_idx]
    else
        annorect = multi_person["annolist"]["annorect"][1, found_idx]
    end
    
    if annorect == nothing 
     throw(DomainError(image_name, "image path has no annorect"))
    end
    
    return reshape([is_single, annorect], (1 ,2))
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
    is_single = indexed_raw_data[2][4]
    annorect = indexed_raw_data[2][5]
    return DataItem(i, img_path, img_size, joints_data, is_single, annorect)
end

function get_from_dataset(dataset, initial_index=1, last_index=max_image_number_to_read; should_use_pmap = true)   
    sizes = dataset["size"][initial_index: last_index]
    images = dataset["image"][initial_index: last_index]
    joints = dataset["joints"][initial_index: last_index]
    is_singles = dataset["is_single"][initial_index: last_index]
    annorects = dataset["annorect"][initial_index: last_index]
    if should_use_pmap
            data_items = pmap(raw_data_to_data_item, enumerate(zip(images, sizes, joints, is_singles, annorects)))
    else
            data_items = map(raw_data_to_data_item, enumerate(zip(images, sizes, joints, is_singles, annorects)))
    end
    return data_items
end

