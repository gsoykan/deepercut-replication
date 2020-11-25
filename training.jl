function trainresults(file, model, data_trn, data_tst, cycle_count, learning_rate, optimizer=sgd, should_save=false; accuracy_func=accuracy, error_func)
    GC.gc(true)
    training = optimizer(model, ncycle(data_trn, cycle_count), lr=learning_rate)
    snapshot() = (deepcopy(model),
     model(data_trn),
      model(data_tst),
      0, 0, 0, 0)
       # compute_error_in_training(model, data_trn, error_func),
         # compute_error_in_training(model, data_tst, error_func), 
         # compute_accuracy_in_training(model, data_trn, accuracy_func),
         # compute_accuracy_in_training(model, data_tst, accuracy_func))
    snapshots = (snapshot() for x in takenth(progress(training), length(data_trn)))
    results = reshape(collect(flatten(snapshots)), (7, :))
    if (should_save) 
        Knet.save(file, "results", results)
    end
    return results

end

function compute_accuracy_in_training(model, data, accuracy_func)
    if accuracy_func !== nothing
        return accuracy_func(model, data)
    else
        return 0
        train_accuracy = 0
    end
end

function compute_error_in_training(model, data, error_func)
    if error_func !== nothing 
        return error_func(model, data)
    else
        return 0 
    end
end
    