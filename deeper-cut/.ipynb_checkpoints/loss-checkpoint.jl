# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
function sigmoid_cross_entropy_loss(scores, labels)
    # max.(y_hat, 0) .- y_hat .* y + log(1 .+ exp(-abs.(y_hat)))
  #   mask = Knet.atype()(zeros(Float32, size(labels)))
   #  mask[:, :, 1:14 ,:] .= 1
    # scores = scores .* mask
    labels =  labels[:, :, 1:14 ,:] # labels .* mask
    l = max.(0, scores) .- labels .* scores .+ log.(1 .+ exp.(-abs.(scores)))
    (sum(l) / length(l)) 
end