using Images, FileIO
using CSV
using ImageTransformations
using Distributed

function read_images(path, resize_image_dim_x, resize_image_dim_y) 
    image_paths = map(path -> "$image_folder_path/$path", filter(path -> contains(path, ".png"), readdir(image_folder_path)))
    loaded_images = map(path -> Float32.(channelview(load(path))), (image_paths))
    channel_size =  size(loaded_images[1])[begin]
    loaded_images = pmap(img -> imresize(img, channel_size, resize_image_dim_x, resize_image_dim_y), loaded_images)
    single_image_size = size(loaded_images[1])
    vcatted_images = vcat(loaded_images...)
    reshape(vcatted_images, (length(image_paths), single_image_size...))
end

function read_labels_as_mask(path, image_x_dim, image_y_dim, number_of_categories)
    labels = []
    for row in CSV.File(path)
        if !contains(row.scorer, ".png")
            continue
        end
        points = values(row)[2:end]
        points = map(e -> parse(Float32, e), points)
        reshaped_points = reshape(points, (number_of_categories, 2))
        label = ones(image_x_dim, image_y_dim)
        for (i, point) in  enumerate(eachrow(reshaped_points))
            rounded_x = convert(Int64, round(round(point[1]), digits=0)) 
            rounded_y = convert(Int64, round(round(point[2]), digits=0)) 
            label[rounded_x, rounded_y] = i + 1
        end
        push!(labels, label)
    end
    vcatted_labels = vcat(labels...)
    reshaped_labels = reshape(vcatted_labels, (length(labels), image_x_dim, image_y_dim))
    return reshaped_labels
end

# compress data between 0 - 1 
function read_labels_as_keypoints(path, image_x_dim, image_y_dim, number_of_categories)
    labels = []
    for row in CSV.File(path)
        if !contains(row.scorer, ".png")
            continue
        end
        points = values(row)[2:end]
        points = map(e -> parse(Float32, e), points)
        
        for (i, point) in enumerate(points)
            points[i] = point / ( isodd(i) ?  image_x_dim : image_y_dim )
        end

        reshaped_points = reshape(points, (number_of_categories * 2, 1))
        push!(labels, reshaped_points)
    end
    vcatted_labels = vcat(labels...)
    reshaped_labels = reshape(vcatted_labels, (length(labels), number_of_categories * 2))
    return reshaped_labels
end

function read_all_data(path,
        image_x_dim,
        image_y_dim,
        number_of_categories; 
        resized_image_dim_x,
        resized_image_dim_y
     )
    image_data = read_images(path, resized_image_dim_x, resized_image_dim_y)
    label_data = read_labels_as_keypoints(path, image_x_dim, image_y_dim, number_of_categories)
    return (image_data, label_data)
end