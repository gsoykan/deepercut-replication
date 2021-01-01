using Knet
include("../helper.jl")
include("mpii.annotation.parser.jl")

function get_mpii_batches_and_data_items(batch_size; should_shuffle = false, should_use_pmap = true)
    dataset = read_cropped_mpii_annotations(; should_shuffle = should_shuffle)
    dtrn = []
    step_size = 1024

    for i = 1:step_size:(train_image_count+1-step_size)
        println(i)
        train_dataset = get_from_dataset(dataset, i, i + step_size - 1; should_use_pmap = should_use_pmap)
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
    
    # TODO: Preparing Validation Data
    validation_dataset = get_from_dataset(
        dataset,
        train_image_count + 1,
        train_image_count + validation_image_count;
        should_use_pmap = should_use_pmap
    )
    validation_preprocessed = preprocess_dataset(validation_dataset)
    dval = get_batch(batch_size, validation_preprocessed; shuffle_in_minibatch = false)

    validation_dataset = 0
    validation_preprocessed = 0
    GC.gc(true)
    
    # Preparing Test Data
    test_dataset = get_from_dataset(
        dataset,
        train_image_count + validation_image_count + 1,
        train_image_count + validation_image_count + test_image_count;
        should_use_pmap = should_use_pmap
    )
    test_preprocessed = preprocess_dataset(test_dataset)
    dtst = get_batch(batch_size, test_preprocessed; shuffle_in_minibatch = false)

    test_dataset = 0
    test_preprocessed = 0
    GC.gc(true)

    # TODO: this might be redundant after all
    data_items = get_from_dataset(dataset, 1, train_image_count + validation_image_count + test_image_count; should_use_pmap = should_use_pmap)
    data_items = map(d_i -> DataItem(d_i, read_image_h, read_image_w), data_items)

    return (dtrn, dval, dtst, data_items)
end

function get_batch(
    batch_size,
    preprocessed;
    shuffle_in_minibatch = true,
    include_scmap_weights = false,
)
    y_scmap = map(element -> element[3], preprocessed)
    y_scmap_weights = map(element -> add_dim(element[4]), preprocessed)
    y_locref_map = map(element -> element[5], preprocessed)
    y_locref_mask = map(element -> element[6], preprocessed)

    y_all = map(
        i -> merge_different_y(
            y_scmap[i],
            y_scmap_weights[i],
            y_locref_map[i],
            y_locref_mask[i];
            include_scmap_weights = include_scmap_weights,
        ),
        1:length(y_scmap),
    )
    x_all = map(element -> element[2], preprocessed)

    xall = cat(x_all..., dims = 4)
    yall = cat(y_all..., dims = 4)

    xsize = size(xall)[1:3]
    ysize = size(yall)[1:3]

    d = minibatch(
        xall,
        yall,
        batch_size;
        xsize = (xsize..., :),
        ysize = (ysize..., :),
        xtype = Knet.atype(),
        shuffle = shuffle_in_minibatch,
    )
    return d
end

function merge_different_y(
    scmap,
    scmap_w,
    locref_map,
    locref_mask;
    include_scmap_weights = false,
)
    if include_scmap_weights == true
        return add_dim(cat(scmap, scmap_w, locref_map, locref_mask; dims = (3)))
    else
        return add_dim(cat(scmap, locref_map, locref_mask; dims = (3)))
    end
end
