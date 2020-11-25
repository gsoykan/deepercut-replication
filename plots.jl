using Plots; default(fmt = :png)

function draw_plots(results)
    trnloss,tstloss = Array{Float32}(results[2,:]), Array{Float32}(results[3,:]) 
    plot_loss = plot([trnloss,tstloss],ylim=(.0,.4),labels=["trnloss" "tstloss"],xlabel="Epochs",ylabel="Loss")

    trnerr,tsterr = Array{Float32}(results[4,:]), Array{Float32}(results[5,:]) 
    plot_error = plot([trnerr,tsterr],ylim=(.0,.12),labels=["trnerr" "tsterr"],xlabel="Epochs",ylabel="Error")

    trnacc,tstacc = Array{Float32}(results[6,:]), Array{Float32}(results[7,:]) 
    plot_acc = plot([trnacc,tstacc],ylim=(.0,1.0),labels=["tracc" "tstacc"],xlabel="Epochs",ylabel="Accuracy")

    foreach(display, [plot_loss, plot_error, plot_acc])  
end