using Knet
# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

function deeper_cut_combined_loss(scores, labels)
    part_detection_labels = labels[:, :, 1:global_num_joints, :]
    part_detection_scores = scores[:, :, 1:global_num_joints, :]
    part_detection_loss =
        sigmoid_cross_entropy_loss(part_detection_scores, part_detection_labels)

    score_channel_size = (scores|>size)[end-1]
    loc_ref_loss = 0
    if score_channel_size >= global_num_joints * 5
        loc_ref_labels = labels[:, :, global_num_joints+1:global_num_joints*3, :]
        loc_ref_scores = scores[:, :, global_num_joints+1:global_num_joints*3, :]

        loc_ref_labels_weights = labels[:, :, global_num_joints*3+1:global_num_joints*5, :]
        loc_ref_scores_weights = scores[:, :, global_num_joints*3+1:global_num_joints*5, :]

        loc_ref_weight_loss =
            sigmoid_cross_entropy_loss(loc_ref_scores_weights, loc_ref_labels_weights)

        loc_ref_loss =
            huber_loss(loc_ref_scores, loc_ref_labels; weights = loc_ref_scores_weights)

        loc_ref_loss = loc_ref_loss + loc_ref_weight_loss
    elseif score_channel_size == global_num_joints * 3
        loc_ref_labels = labels[:, :, global_num_joints+1:global_num_joints*3, :]
        loc_ref_scores = scores[:, :, global_num_joints+1:global_num_joints*3, :]
        loc_ref_loss = huber_loss(loc_ref_scores, loc_ref_labels)
    end

    return part_detection_loss + locref_loss_weight * loc_ref_loss
end

function sigmoid_cross_entropy_loss(scores, labels)
    # max.(y_hat, 0) .- y_hat .* y + log(1 .+ exp(-abs.(y_hat)))
    #   mask = Knet.atype()(zeros(Float32, size(labels)))
    #  mask[:, :, 1:14 ,:] .= 1
    # scores = scores .* mask
    # labels .* mask  
    l = max.(0, scores) .- labels .* scores .+ log.(1 .+ exp.(-abs.(scores)))
    (sum(l) / length(l))
end

function huber_loss(scores, labels; weights = 1, beta = 1)
    diff = abs.(labels .- scores)
    cpu_diff = Array(diff)
    low_idx = findall(cpu_diff .< beta)
    high_idx = findall(cpu_diff .>= beta)

    # TODO: put weights into sigmoids
    sigmoided_weights = weights == 1 ? Knet.atype()(ones(size(scores))) : sigm.(weights)

    loss_sum = sum((diff[high_idx] .- 0.5 .* beta^2) .* sigmoided_weights[high_idx])
    loss_sum += sum((0.5 .* (diff[low_idx] .^ 2) .* sigmoided_weights[low_idx]))
    return sum(loss_sum) / length(diff)
end

#=
def huber_loss(labels, predictions, weight=1.0, k=1.0, scope=None):
    """Define a huber loss  https://en.wikipedia.org/wiki/Huber_loss
      tensor: tensor to regularize.
      k: value of k in the huber loss
      scope: Optional scope for op_scope.

    Huber loss:
    f(x) = if |x| <= k:
              0.5 * x^2
           else:
              k * |x| - 0.5 * k^2

    Returns:
      the L1 loss op.

    http://concise-bio.readthedocs.io/en/latest/_modules/concise/tf_helper.html
    """
    with ops.name_scope(scope, "absolute_difference",
                        [predictions, labels]) as scope:
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        if weight is None:
            raise ValueError("`weight` cannot be None")
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        diff = math_ops.subtract(predictions, labels)
        abs_diff = tf.abs(diff)
        losses = tf.where(abs_diff < k,
                          0.5 * tf.square(diff),
                          k * abs_diff - 0.5 * k ** 2)
        return tf.losses.compute_weighted_loss(losses, weight)

=#
