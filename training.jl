function trainresults(
    save_tag,
    model,
    data_trn,
    data_tst,
    learning_rate_per_epoch,
    optimizer = sgd,
    should_save = false;
    accuracy_func = accuracy,
    error_func,
    secondary_accuracy_func,
    data_items_trn,
    data_items_val,
    check_loss = true
)
    GC.gc(true)
    epoch_count = 0
    # TODO: we can make better use of it no need to run forward pass multiple times
    function snapshot(i)

        trn_loss = check_loss ? model(data_trn) : 0
        tst_loss = check_loss ? model(data_trn) : 0

        naive_trn_accuracy = compute_accuracy_in_training(model, data_trn, accuracy_func)
        naive_tst_accuracy = compute_accuracy_in_training(model, data_tst, accuracy_func)

        pck_trn_accuracy, pck_trn_distribution =
            data_items_trn == nothing ? (0, 0) :
            compute_secondary_accuracy_in_training(
                model,
                data_trn,
                data_items_trn,
                secondary_accuracy_func,
            )

        pck_val_accuracy, pck_val_distribution = data_items_val == nothing ?  (0, 0) : compute_secondary_accuracy_in_training(
            model,
            data_tst,
            data_items_val,
            secondary_accuracy_func,
        )

        if pck_val_distribution != 0 
                pck_val_distribution = read_accuracy_results(pck_val_distribution)
        end
    
        open("$(save_tag)-training_snapshots.txt", "a") do io
            write(io, "***** epoch: $(epoch_count) ***** \n")
            write(io, "trn_loss: $(trn_loss) \n")
            write(io, "tst_loss: $(tst_loss) \n")
            write(io, "naive_trn_accuracy: $(naive_trn_accuracy) \n")
            write(io, "naive_tst_accuracy: $(naive_tst_accuracy)  \n")
            write(io, "pck_val_accuracy: $(pck_val_accuracy)  \n")
            write(io, "$(pck_val_distribution)")
            write(io, "\n")
        end

        epoch_count += 1

        Knet.save("$(save_tag)-training_model.jld2", "$(save_tag)-training_model", model)

        snap_res = (
            0,
            trn_loss,
            tst_loss,
            0,
            0,
            naive_trn_accuracy,
            naive_tst_accuracy,
            pck_trn_accuracy,
            pck_val_accuracy,
        )
        println(snap_res)
        return snap_res
    end
    # compute_error_in_training(model, data_trn, error_func),
    # compute_error_in_training(model, data_tst, error_func), 

    results = []
    for (lr, for_epoch) in learning_rate_per_epoch
        training = optimizer(model, ncycle(data_trn, for_epoch), lr = lr)

        #   (snapshot(x) for x in takenth(progress(training), length(data_trn))) |> collect

        snapshots = (snapshot(x) for x in takenth(progress(training), length(data_trn)))
        intermediate_res = reshape(collect(flatten(snapshots)), (9, :))
        if isempty(results)
            results = intermediate_res
        else
            results = hcat(results, intermediate_res)
        end

    end

    if (should_save)
        Knet.save("$(save_tag)_results.jld2", "$(save_tag)_results", results)
    end
    return results

end

# TODO: modify this for our purposes
function alternative_train!(model, trn, dev, tst...; iteration_count = 1000)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), steps = iteration_count) do y
        losses = [loss(model, d) for d in (dev, tst...)]
        if losses[1] < bestloss
            bestmodel, bestloss = deepcopy(model), losses[1]
        end
        return (losses...,)
    end
    return bestmodel
end

function compute_accuracy_in_training(model, data, accuracy_func)
    if accuracy_func !== nothing
        return accuracy_func(model, data)
    else
        return 0
        train_accuracy = 0
    end
end

function compute_secondary_accuracy_in_training(model, data, data_items, accuracy_func)
    if accuracy_func !== nothing
        return accuracy_func(model, data, data_items)
    else
        return (0, 0)
    end
end

function compute_error_in_training(model, data, error_func)
    if error_func !== nothing
        return error_func(model, data)
    else
        return 0
    end
end
