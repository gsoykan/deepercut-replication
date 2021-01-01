using Knet
include("../helper.jl")
include("accuracy.jl")
include("loss.jl")
include("mpii.batcher.jl")
include("../models.jl")
include("../training.jl")
include("../plots.jl")
include("../modular.resnet.jl")

test_image_count = 1000
validation_image_count =  28883 - 25600 - 1000
train_image_count = 25600
read_image_w = 512
read_image_h = 512
pre_full_path = "/kuacc/users/gsoykan20/comp541_term_project/deeper-cut/"

function load_and_save_batches(save_tag; should_use_pmap = true)
    dtrn, dval, dtst, data_items = get_mpii_batches_and_data_items(1; should_shuffle=true, should_use_pmap = should_use_pmap);
trn_data_items = data_items[begin : end - validation_image_count - test_image_count];
val_data_items = data_items[end - validation_image_count - test_image_count + 1 : end - test_image_count]
tst_data_items = data_items[end - test_image_count + 1 : end];

    println("start saving dtrn")
    Knet.save("$(pre_full_path)$(save_tag)-dtrn.jld2", "$(save_tag)-dtrn", dtrn)

    println("start saving dval")
    Knet.save("$(pre_full_path)$(save_tag)-dval.jld2", "$(save_tag)-dval", dval)
    
    println("start saving dtst")
    Knet.save("$(pre_full_path)$(save_tag)-dtst.jld2", "$(save_tag)-dtst", dtst)

    println("start saving trn data items")
    Knet.save("$(pre_full_path)$(save_tag)-trn_data_items.jld2", "$(save_tag)-trn_data_items", trn_data_items)

    println("start saving val data items")
    Knet.save("$(pre_full_path)$(save_tag)-val_data_items.jld2", "$(save_tag)-val_data_items", val_data_items)
    
    println("start saving tst data items")
    Knet.save("$(pre_full_path)$(save_tag)-tst_data_items.jld2", "$(save_tag)-tst_data_items", tst_data_items)

    println("save completed!")
    return (dtrn, dval, dtst, trn_data_items, val_data_items, tst_data_items)
end
