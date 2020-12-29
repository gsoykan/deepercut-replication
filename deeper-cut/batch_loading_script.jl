using Knet
include("../helper.jl")
include("accuracy.jl")
include("loss.jl")
include("mpii.batcher.jl")
include("../models.jl")
include("../training.jl")
include("../plots.jl")
include("../modular.resnet.jl")

validation_image_count = 28883 - 25600
train_image_count = 25600
read_image_w = 512
read_image_h = 512

function load_and_save_batches()
    dtrn, dval, data_items = get_mpii_batches_and_data_items(1; should_shuffle = true)
    trn_data_items = data_items[begin:end-validation_image_count]
    val_data_items = data_items[end-validation_image_count+1:end]

    println("start saving dtrn")
    Knet.save("29-12-20-dtrn.jld2", "29-12-20-dtrn", dtrn)

    println("start saving dval")
    Knet.save("29-12-20-dval.jld2", "29-12-20-dval", dval)

    println("start saving trn data items")
    Knet.save("29-12-20-trn_data_items.jld2", "29-12-20-trn_data_items", trn_data_items)

    println("start saving dtrn")
    Knet.save("29-12-20-val_data_items.jld2", "29-12-20-val_data_items", val_data_items)

    println("save completed!")
end
