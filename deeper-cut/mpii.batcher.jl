using Knet
include("../helper.jl")
include("mpii.annotation.parser.jl")

function get_mpii_batches_and_data_items(batch_size; should_shuffle=false)
    dataset = read_cropped_mpii_annotations(;should_shuffle=should_shuffle)
    dtrn = []
    step_size = 128
    for i in 1:step_size:(train_image_count + 1 - step_size)
        println(i)
        train_dataset = get_from_dataset(dataset, i, i + step_size - 1)
        if isempty(train_dataset)
            println("continued")
            continue
        end
        train_preprocessed = preprocess_dataset(train_dataset)
        dtrn_part = get_batch(batch_size, train_preprocessed)
        if isempty(dtrn)
            dtrn = dtrn_part
            println("$(summary(dtrn))")
        else
            append_to_data!(dtrn, dtrn_part)
            println("$(summary(dtrn))")
        end
        train_dataset = 0
        train_preprocessed = 0
        GC.gc(true)
    end
    
    # TODO: There might be small intersection between validation and train data
    
    validation_dataset = get_from_dataset(dataset, train_image_count + 1, train_image_count + validation_image_count)
    validation_preprocessed = preprocess_dataset(validation_dataset)
    dval = get_batch(batch_size, validation_preprocessed; shuffle_in_minibatch=false)    
    
    validation_dataset = 0
    validation_preprocessed = 0
    GC.gc(true)
    
    data_items = get_from_dataset(dataset, 1, train_image_count + validation_image_count)
    data_items = map(d_i -> DataItem(d_i, read_image_h, read_image_w), data_items)

    return  (dtrn, dval, data_items)
end

function get_batch(batch_size, preprocessed; shuffle_in_minibatch=true)
    y_scmap = map(element -> element[3], preprocessed);
    y_scmap_weights = map(element -> add_dim(element[4]), preprocessed);
    y_locref_map = map(element -> element[5], preprocessed);
    y_locref_mask = map(element -> element[6], preprocessed);
    
    y_all = map(i ->  merge_different_y(y_scmap[i], y_scmap_weights[i], y_locref_map[i], y_locref_mask[i]), 1:length(y_scmap));
    x_all = map(element -> element[2], preprocessed);
    
    xall = cat(x_all..., dims=4);
    yall = cat(y_all..., dims=4);

    xsize = size(xall)[1:3]
    ysize = size(yall)[1:3]

    d = minibatch(xall,
        yall,
        batch_size;
        xsize=(xsize..., :),
        ysize=(ysize..., :), 
        xtype=Knet.atype(), 
        shuffle=shuffle_in_minibatch)
    return d
end

function merge_different_y(scmap, scmap_w, locref_map, locref_mask)
    return add_dim(cat(scmap,  
     scmap_w,
     locref_map,
     locref_mask; dims=(3)));
end