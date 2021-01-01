using Knet
include("../helper.jl")
include("accuracy.jl")
include("loss.jl")
include("mpii.batcher.jl")
include("../models.jl")
include("../training.jl")
include("../plots.jl")
include("../modular.resnet.jl")
include("batch_loading_script.jl")

# example usage
# continue_training("29-12-20", "29-12-20-cont", [(0.002, 10), (0.001, 12)])

pre_full_path = "/kuacc/users/gsoykan20/comp541_term_project/deeper-cut/"

function init_training(; save_tag, learning_rate_by_epochs, should_use_pmap = true, check_loss = false)
    dtrn, dval, dtst, trn_data_items, val_data_items, tst_data_items = load_and_save_batches(save_tag; should_use_pmap = should_use_pmap)

    init_training_with_resnet152(
        dtrn,
        dval,
        val_data_items;
        save_tag = save_tag,
        learning_rate_by_epochs = learning_rate_by_epochs,
        check_loss = check_loss
    )
end

function init_training_with_resnet152(
    dtrn,
    dval,
    val_data_items;
    save_tag,
    learning_rate_by_epochs,
    check_loss = true
)
    CUDA.reclaim()
    CUDA.memory_status()
    GC.gc(true)
    println("cleared gpu memory")
    # Uses ResNet 152
    deeper_cut_with_loc_ref =
        generate_deeper_cut(; should_use_resnet50 = false, is_loc_ref_enabled = true)
    println("prepared deepercut")

    CUDA.reclaim()
    CUDA.memory_status()
    GC.gc(true)
    println("cleared gpu memory")

    # [(0.005, 1), (0.02, 17), (0.002, 12), (0.001, 12)]
    println("started training")
    deeper_cut_with_loc_ref_results = trainresults(
        save_tag,
        deeper_cut_with_loc_ref,
        dtrn,
        dval,
        learning_rate_by_epochs;
        accuracy_func = nothing,
        error_func = nothing,
        secondary_accuracy_func = modelized_PCKh_sigm,
        data_items_trn = nothing,
        data_items_val = val_data_items,
        check_loss = check_loss
    )
end

# learning_rate_by_epochs: [(0.005, 1), (0.02, 17), (0.002, 12), (0.001, 12)]
function continue_training(save_tag, new_save_tag, learning_rate_by_epochs)
    println("start loading dtrn")
    dtrn = Knet.load("$(pre_full_path)$(save_tag)-dtrn.jld2", "$(save_tag)-dtrn")

    println("start loading dval")
    dval = Knet.load("$(pre_full_path)$(save_tag)-dval.jld2", "$(save_tag)-dval")

    println("start loading val data items")
    val_data_items =
        Knet.load("$(pre_full_path)$(save_tag)-val_data_items.jld2", "$(save_tag)-val_data_items")

    println("start loading model")
    model = Knet.load("$(pre_full_path)$(save_tag)-training_model.jld2", "$(save_tag)-training_model")

    CUDA.reclaim()
    CUDA.memory_status()
    GC.gc(true)
    println("cleared gpu memory")

    trainresults(
        new_save_tag,
        model,
        dtrn,
        dval,
        learning_rate_by_epochs;
        accuracy_func = nothing,
        error_func = nothing,
        secondary_accuracy_func = modelized_PCKh_sigm,
        data_items_trn = nothing,
        data_items_val = val_data_items,
    )

end
