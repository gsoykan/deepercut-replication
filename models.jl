using Knet:
    Knet,
    AutoGrad,
    dir,
    Data,
    minibatch,
    Param,
    @diff,
    value,
    params,
    grad,
    progress,
    progress!,
    KnetArray
using IterTools
using Base.Iterators
using Knet: Ops20.zeroone
using Printf, Random, Test, Statistics
using Plots;
default(fmt = :png);
include("utils.jl")
include("./deeper-cut/variant.data.jl")

# Define Linear Model
struct Linear
    w::Any
    b::Any
end
Linear(i::Int, o::Int) = Linear(param(o, i), param0(o))
(m::Linear)(x) = m.w * x .+ m.b
(m::Linear)(x, y) = Knet.nll(m(x), y)
(m::Linear)(data::Knet.Data) = mean(m(x, y) for (x, y) in data)

# MLP Layer - f: can be identity, relu etc,  - pdrop: dropout rate
struct MLPLayer
    w::Any
    b::Any
    f::Any
    pdrop::Any
end
MLPLayer(i::Int, o::Int, f = relu; pdrop = 0) = MLPLayer(param(o, i), param0(o), f, pdrop)
(l::MLPLayer)(x) = l.f.(l.w * dropout(x, l.pdrop) .+ l.b)

# Define convolutional layer:
struct Conv2
    w::Any
    b::Any
    f::Any
    is_pool_enabled::Bool
end
Conv2(w1, w2, nx, ny, f = relu; is_pool_enabled = true) =
    Conv2(param(w1, w2, nx, ny), param0(1, 1, ny, 1), f, is_pool_enabled)
function (c::Conv2)(x)
    if c.is_pool_enabled
        pool(c.f.(conv4(c.w, x) .+ c.b))
    else
        c.f.(conv4(c.w, x) .+ c.b)
    end
end

# Deconv
struct Deconv
    w::Any
    b::Any
    stride::Any
    padding::Any
    tag::String
end
Deconv(w1, w2, nx, ny; stride = 1, padding = 0, atype = Knet.atype(), tag = "") =
    Deconv(param(w1, w2, nx, ny; atype = atype), param0(1, 1, nx, 1), stride, padding, tag)

Deconv(w, b; stride = 1, padding = 0, atype = Knet.atype(), tag = "") =
    Deconv(param(w; atype = atype), param(b; atype = atype), stride, padding, tag)

function (dc::Deconv)(x)
    dc_res = deconv4(dc.w, x; stride = dc.stride, padding = dc.padding)
        
 #=   println("deconv - dc res")
    println(sum(isnan.(Array(dc_res))))
    if ( (sum(isnan.(Array(dc_res))) > 0 ))
          global  buggy_w = dc.w
    global buggy_x = x
    global buggy_stride = dc.stride
    global buggy_padding = dc.padding
    end=#
    
    res = dc_res .+ dc.b    
    return res
end

function get_weights_sum(l::Deconv)
    return sum(abs2, l.w)
end

# Define dense layer:
struct Dense
    w::Any
    b::Any
    f::Any
    p::Any
end
Dense(i::Int, o::Int, f = relu; pdrop = 0) = Dense(param(o, i), param0(o), f, pdrop)
Dense(w, b; f = identity, pdrop = 0) =
    Dense(param(w; atype = Knet.atype()), param(b; atype = Knet.atype()), f, pdrop)
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b)

# Define a chain of layers and a loss function:
struct DeeperCutOption
    connect_res3_to_res5::Bool

    function DeeperCutOption(; connect_res3_to_res5 = false)
        return new(connect_res3_to_res5)
    end
end

# Deeper Cut Head
struct DeeperCutHead
    part_detection_head::Deconv
    loc_ref_head::Deconv
    tag::String
    is_loc_ref_enabled::Bool

    function DeeperCutHead(;
        part_detection_head = Deconv(
            4,
            4,
            global_num_joints,
            2048;
            padding = 1,
            stride = 2,
            tag = "part_detect_deconv",
        ),
        loc_ref_head = Deconv(
            4,
            4,
            global_num_joints * 4,
            2048;
            padding = 1,
            stride = 2,
            tag = "loc_ref_deconv",
        ),
        tag = "deeper_cut_head",
        is_loc_ref_enabled = false,
    )
        return new(part_detection_head, loc_ref_head, tag, is_loc_ref_enabled)
    end

    function DeeperCutHead(
        p_d_w,
        p_d_b,
        l_r_w,
        l_r_b;
        tag = "deeper_cut_head",
        is_loc_ref_enabled = false,
    )
        part_detection_head =
            Deconv(p_d_w, p_d_b; padding = 1, stride = 2, tag = "part_detect_deconv")
        loc_ref_head = Deconv(l_r_w, l_r_b; padding = 1, stride = 2, tag = "loc_ref_deconv")
        return new(part_detection_head, loc_ref_head, tag, is_loc_ref_enabled)
    end

end

function (deeper_cut_head::DeeperCutHead)(x)
    # TODO: how shall we check if the x is intact between heads
    part_detection_result = deeper_cut_head.part_detection_head(x)    

    if deeper_cut_head.is_loc_ref_enabled == true
        loc_ref_result = deeper_cut_head.loc_ref_head(x)
        
        channel_dim = (size(part_detection_result) |> length) - 1
        combined_result = cat(part_detection_result, loc_ref_result; dims = channel_dim)
        return combined_result
    else
        return part_detection_result
    end
end

function get_weights_sum(l::DeeperCutHead)
    # sum(abs2, l.w)
    sum1 =  get_weights_sum(l.part_detection_head)
    sum2 =  get_weights_sum(l.loc_ref_head)
    return sum1 + sum2 
end

# Chain
struct Chain
    layers::Any
    lambda1::Any
    lambda2::Any
    loss::Any
    deeperCutOption::Any
    Chain(layers...; lambda1 = 0, lambda2 = 0, loss = nll, deeperCutOption = nothing) =
        new(layers, lambda1, lambda2, loss, deeperCutOption)
end

# The prediction and average loss do not change
function (c::Chain)(x)
    connection_from3_to5 = nothing
    connection_from3_to5_loc_ref = nothing
        
    for l in c.layers
        x = l(x)
    
        if c.deeperCutOption != nothing && c.deeperCutOption.connect_res3_to_res5
            layer_tag = get_object_tag(l)
            if layer_tag == 3
                connection_from3_to5 = l.conv3_for_deepercut_output(x)
                connection_from3_to5_loc_ref = l.conv3_for_deepercut_output_loc_ref(x)
            end
        end

        if c.deeperCutOption != nothing && c.deeperCutOption.connect_res3_to_res5
            layer_tag = get_object_tag(l)
            if layer_tag == "deeper_cut_head"
                if l.is_loc_ref_enabled == true
                    channel_dim = (size(connection_from3_to5) |> length) - 1
                    combined_result = cat(
                        connection_from3_to5,
                        connection_from3_to5_loc_ref;
                        dims = channel_dim,
                    )
                    # TODO: This introduces problems when the dimensions are odd
                    x = x .+ combined_result
                else
                    x = x .+ connection_from3_to5
                end
            end
        end

    end

    x
end

function (c::Chain)(x, y)
    loss = c.loss(c(x), y)
    if training() # Only apply regularization during training, only to weights, not biases.
        c.lambda1 != 0 && (loss += c.lambda1 * sum(sum(abs, l.w) for l in c.layers))
        c.lambda2 != 0 && (loss += c.lambda2 * sum( get_weights_sum(l) for l in c.layers))
    end
    return loss
end
(c::Chain)(d::Data) = mean(c(x, y) for (x, y) in d)
(c::Chain)(d::VariantData) = mean(c(x, y) for (x, y) in d)

# This became redundant now
struct ResLayerConv
    w::Any
    padding::Any
    stride::Any
end
# Random init
ResLayerConv(w1, w2, nx, ny; padding = 0, stride = 1) =
    ResLayerConv(param(w1, w2, nx, ny), padding, stride)
# Predeterminde weights
ResLayerConv(w; padding = 0, stride = 1) = ResLayerConv(param(w), padding, stride)
(rl0::ResLayerConv)(x) = conv4(rl0.w, x; padding = rl0.padding, stride = rl0.stride)

struct BatchNormLayer
    w::Any
    ms::Any

    function BatchNormLayer(pre_w, pre_ms; freeze = false)
        res_mean = popfirst!(pre_ms)
        # Trick to arrange variance value for new(er) batchnorm
        
        # TODO: DO NOT FORGET TO ADD IT
        
        res_variance = popfirst!(pre_ms) .^ 2 # .- 1e-5
        ms = bnmoments(mean = res_mean, var = res_variance)

        w1 = pre_w[1]
        w2 = pre_w[2]
        w1 = vec(w1)
        w2 = vec(w2)
        w = vcat(w1, w2)
        if freeze
            return new(w, ms)
        else
            param_w = param(w, atype = Knet.atype())
            return new(param_w, ms)
        end
    end

end

function (batch_norm_layer::BatchNormLayer)(x)
    return batchnorm(x, batch_norm_layer.ms, batch_norm_layer.w; eps = 1e-5)
end

# ResNet 50 initial layer
struct ResLayerX1_50
    batch_layer::Any
    conv_w::Any
    conv_b::Any
    padding::Any
    stride::Any
    pool_window_size::Any
    pool_stride::Any
    pool_padding::Any
end

function ResLayerX1_50(
    w,
    ms;
    padding = 3,
    stride = 2,
    pool_window_size = 3,
    pool_stride = 2,
    pool_padding = 1,
    freeze_batchnorm = true,
)
    bnl = BatchNormLayer(w[3:4], ms; freeze = freeze_batchnorm)
    return ResLayerX1_50(
        bnl,
        param(w[1]; atype = Knet.atype()),
        param(w[2]; atype = Knet.atype()),
        padding,
        stride,
        pool_window_size,
        pool_stride,
        pool_padding,
    )
end

function (rlx1_50::ResLayerX1_50)(x)
    o =
        conv4(rlx1_50.conv_w, x; padding = rlx1_50.padding, stride = rlx1_50.stride) .+
        rlx1_50.conv_b
    o = rlx1_50.batch_layer(o)
    o = relu.(o)
    o = pool(
        o;
        window = rlx1_50.pool_window_size,
        stride = rlx1_50.pool_stride,
        padding = rlx1_50.pool_padding,
    )
    return o
end

function get_weights_sum(l::ResLayerX1_50)
    # sum(abs2, l.w)
    sum1 = sum(abs2, l.batch_layer.w)
    sum2 = sum(abs2, l.conv_w)
    return sum1 + sum2 
end

# X0
struct ResLayerX0
    batch_layer::Any
    conv_w::Any
    padding::Any
    stride::Any
    dilation::Any
end
# Predetermined weights
# TODO: should we try to make bnl params??
function ResLayerX0(w, ms; padding = 0, stride = 1, dilation = 1, freeze_batchnorm = true)
    bnl = BatchNormLayer(w[2:3], ms; freeze = freeze_batchnorm)
    return ResLayerX0(bnl, param(w[1]; atype = Knet.atype()), padding, stride, dilation)
end

function (rlx0::ResLayerX0)(x)
    # batchnorm_as_function(rlx0.batch_w, conv4(rlx0.conv_w, x; padding=rlx0.padding, stride=rlx0.stride), rlx0.ms) 
    o = conv4(
        rlx0.conv_w,
        x;
        padding = rlx0.padding,
        stride = rlx0.stride,
        dilation = rlx0.dilation,
    )
    o = rlx0.batch_layer(o)
    return o
end

function get_weights_sum(l::ResLayerX0)
    # sum(abs2, l.w)
    sum1 = sum(abs2, l.batch_layer.w)
    sum2 = sum(abs2, l.conv_w)
    return sum1 + sum2 
end

# X1
struct ResLayerX1
    x0_layer::Any
    is_initial::Bool
end
ResLayerX1(w, ms; padding = 0, stride = 1, is_initial::Bool = false, dilation = 1) =
    ResLayerX1(
        ResLayerX0(w, ms; padding = padding, stride = stride, dilation = dilation),
        is_initial,
    )
function (rlx1::ResLayerX1)(x)
    relu_res = relu.(rlx1.x0_layer(x))
    if rlx1.is_initial
        return pool(relu_res; window = 3, stride = 2, padding = 1)
    else
        return relu_res
    end
end

function get_weights_sum(l::ResLayerX1)
    # sum(abs2, l.w)
    return get_weights_sum(l.x0_layer)
end


# X2
# TODO: can be constructed like Chain
struct ResLayerX2
    x1_a_layer::Any
    x1_b_layer::Any
    x0_c_layer::Any
end
ResLayerX2(w, ms; pads = [0, 1, 0], strides = [1, 1, 1], dilations = [1, 1, 1]) =
    ResLayerX2(
        ResLayerX1(
            w[1:3],
            ms;
            padding = pads[1],
            stride = strides[1],
            dilation = dilations[1],
        ),
        ResLayerX1(
            w[4:6],
            ms;
            padding = pads[2],
            stride = strides[2],
            dilation = dilations[2],
        ),
        ResLayerX0(
            w[7:9],
            ms;
            padding = pads[3],
            stride = strides[3],
            dilation = dilations[3],
        ),
    )
(rlx2::ResLayerX2)(x) = rlx2.x0_c_layer(rlx2.x1_b_layer((rlx2.x1_a_layer(x))))

function get_weights_sum(l::ResLayerX2)
    # sum(abs2, l.w)
    sum1 = get_weights_sum(l.x1_a_layer) 
    sum2 = get_weights_sum(l.x1_b_layer)
    sum3 = get_weights_sum(l.x0_c_layer)
    return sum1 + sum2 + sum3
end

# X3
struct ResLayerX3
    x0_a_layer::ResLayerX0
    x2_b_layer::ResLayerX2
end
ResLayerX3(
    w,
    ms;
    pads = [0, 0, 1, 0],
    strides = [2, 2, 1, 1],
    b_layer_dilations = [1, 1, 1],
) = ResLayerX3(
    ResLayerX0(w[1:3], ms; padding = pads[1], stride = strides[1]),
    ResLayerX2(
        w[4:12],
        ms;
        pads = pads[2:4],
        strides = strides[2:4],
        dilations = b_layer_dilations,
    ),
)
function (rlx3::ResLayerX3)(x)
    res_a = rlx3.x0_a_layer(x)
    res_b = rlx3.x2_b_layer(x)
    return relu.(res_a .+ res_b)
end

function get_weights_sum(l::ResLayerX3)
    # sum(abs2, l.w)
    sum1 = get_weights_sum(l.x0_a_layer) 
    sum2 = get_weights_sum(l.x2_b_layer)
    return sum1 + sum2
end

# X4
struct ResLayerX4
    x2_layer::Any
end
ResLayerX4(w, ms; pads = [0, 1, 0], strides = [1, 1, 1], dilations = [1, 1, 1]) =
    ResLayerX4(ResLayerX2(w, ms; pads = pads, strides = strides, dilations = dilations))
(rlx4::ResLayerX4)(x) = relu.(x .+ rlx4.x2_layer(x))

function get_weights_sum(l::ResLayerX4)
    # sum(abs2, l.w)
    sum1 = get_weights_sum(l.x2_layer) 
    return sum1
end

# X5
struct ResLayerX5
    x3_layer::ResLayerX3
    x4_layers::Any
    is_next_fc::Bool
    conv3_for_deepercut_output::Any
    conv3_for_deepercut_output_loc_ref::Any
    tag::Any
end


function ResLayerX5(
    w,
    ms;
    strides = [2, 2, 1, 1],
    is_next_fc::Bool = false,
    b_layer_dilations = [1, 1, 1],
    b_layer_pads = [0, 1, 0],
    is_conv3_for_deepercut = false,
    tag = nothing,
)
    x3_layer::ResLayerX3 = ResLayerX3(w[1:12], ms; strides = strides)
    x4_layers = []
    for k = 13:9:length(w)
        layer = ResLayerX4(w[k:k+8], ms; dilations = b_layer_dilations, pads = b_layer_pads)
        push!(x4_layers, layer)
    end

    conv3_for_deepercut_output = nothing
    conv3_for_deepercut_output_loc_ref = nothing
    if is_conv3_for_deepercut
        # 14 layer for part detection
        conv3_for_deepercut_output =
            Conv2(1, 1, 512, global_num_joints, identity; is_pool_enabled = false)
        conv3_for_deepercut_output_loc_ref =
            Conv2(1, 1, 512, global_num_joints * 4, identity; is_pool_enabled = false)
    end

    return ResLayerX5(
        x3_layer,
        x4_layers,
        is_next_fc,
        conv3_for_deepercut_output,
        conv3_for_deepercut_output_loc_ref,
        tag,
    )
end

function (rlx5::ResLayerX5)(x)
    x = rlx5.x3_layer(x)
    for l in rlx5.x4_layers
        x = l(x)
    end

    if rlx5.is_next_fc
        return pool(x; stride = 1, window = 7, mode = 2)
    else
        return x
    end
end

function get_weights_sum(l::ResLayerX5)
    # sum(abs2, l.w)
    sum = get_weights_sum(l.x3_layer) 
      for layer in l.x4_layers
       sum += get_weights_sum(layer)
    end    
    return sum
end
