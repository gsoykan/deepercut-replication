function trainresults(file, model, data_trn, data_tst, learning_rate_per_epoch, optimizer=sgd, should_save=false; accuracy_func=accuracy, 
        error_func,
        secondary_accuracy_func,
        data_items_trn,
        data_items_val)
    GC.gc(true)
    
    function snapshot(i)
       return (0,
     model(data_trn),
      model(data_tst),
      0, 0, 
          compute_accuracy_in_training(model, data_trn, accuracy_func),
          compute_accuracy_in_training(model, data_tst, accuracy_func),
            compute_secondary_accuracy_in_training(model, data_trn, data_items_trn, secondary_accuracy_func),
            compute_secondary_accuracy_in_training(model, data_tst, data_items_val, secondary_accuracy_func)
            ) 
    end
       # compute_error_in_training(model, data_trn, error_func),
         # compute_error_in_training(model, data_tst, error_func), 
        
    results = []
    for (lr, for_epoch) in learning_rate_per_epoch
            training = optimizer(model, ncycle(data_trn, for_epoch), lr=lr)
    snapshots = (snapshot(x) for x in takenth(progress(training), length(data_trn))) 
     intermediate_res = reshape(collect(flatten(snapshots)), (9, :))
        if isempty(results)
            results = intermediate_res
            else 
        results = hcat(results, intermediate_res)
        end
    end
     
   
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

function compute_secondary_accuracy_in_training(model, data, data_items, accuracy_func)
    if accuracy_func !== nothing
        return accuracy_func(model, data, data_items)
    else
        return 0
    end
end

function compute_error_in_training(model, data, error_func)
    if error_func !== nothing 
        return error_func(model, data)
    else
        return 0 
    end
end
    