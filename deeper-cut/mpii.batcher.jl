using Knet
include("../helper.jl")
include("mpii.annotation.parser.jl")

function get_mpii_batches(batch_size)
    dataset = get_dataset()
    preprocessed = preprocess_dataset(dataset)

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

    xtrn = xall[:,:,:, begin:end - validation_image_count]
    ytrn = yall[:,:,:, begin:end - validation_image_count]

    xval = xall[:,:,:, end - validation_image_count + 1:end]
    yval = yall[:,:,:, end - validation_image_count + 1:end]

    dtrn = minibatch(xtrn, ytrn, batch_size; xsize=(xsize..., :), ysize=(ysize..., :), xtype=Knet.atype())
    dval =  minibatch(xval, yval, batch_size; xsize=(xsize..., :), ysize=(ysize..., :), xtype=Knet.atype())
    return (dtrn, dval)
end

function merge_different_y(scmap, scmap_w, locref_map, locref_mask)
    return add_dim(cat(scmap,  
     scmap_w,
     locref_map,
     locref_mask; dims=(3)));
end