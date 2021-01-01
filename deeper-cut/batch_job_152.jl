include("training_script.jl")

init_training(; save_tag="31-12-20_resnet152", learning_rate_by_epochs = [(0.005, 1), (0.02, 17), (0.002, 12), (0.001, 12)], should_use_pmap = true)
