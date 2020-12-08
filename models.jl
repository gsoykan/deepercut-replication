using Knet: Knet, AutoGrad, dir, Data, minibatch, Param, @diff, value, params, grad, progress, progress!, KnetArray
using IterTools
using Base.Iterators
using Knet: Ops20.zeroone
using Printf, Random, Test, Statistics
using Plots; default(fmt=:png)
include("utils.jl")

# Define Linear Model
struct Linear; w; b; end
Linear(i::Int,o::Int) = Linear(param(o, i), param0(o))
(m::Linear)(x) = m.w * x .+ m.b
(m::Linear)(x, y) = Knet.nll(m(x), y)
(m::Linear)(data::Knet.Data) = mean(m(x, y) for (x, y) in data)

# MLP Layer - f: can be identity, relu etc,  - pdrop: dropout rate
struct MLPLayer; w; b; f; pdrop; end
MLPLayer(i::Int,o::Int,f=relu; pdrop=0) = MLPLayer(param(o, i), param0(o), f, pdrop)
(l::MLPLayer)(x) = l.f.(l.w * dropout(x, l.pdrop) .+ l.b)

# Define convolutional layer:
struct Conv2; w; b; f; end
Conv2(w1,w2,nx,ny, f=relu) = Conv2(param(w1, w2, nx, ny), param0(1, 1, ny, 1), f)
(c::Conv2)(x) = pool(c.f.(conv4(c.w, x) .+ c.b))

# Define dense layer:
struct Dense; w; b; f; p; end
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o, i), param0(o), f, pdrop)
Dense(w, b; f=identity, pdrop=0) = Dense(param(w; atype=Knet.atype()), param(b; atype=Knet.atype()), f, pdrop)
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) 

# Define a chain of layers and a loss function:
struct Chain
    layers; lambda1; lambda2; loss;
    Chain(layers...; lambda1=0, lambda2=0, loss=nll) = new(layers, lambda1, lambda2, loss)
end

struct Deconv; w; stride; padding; end
Deconv(w1, w2, nx, ny; stride=1, padding=0) = Deconv(param(w1, w2, nx, ny), stride, padding)
(dc::Deconv)(x) = deconv4(dc.w, x; stride=dc.stride, padding=dc.padding)

# The prediction and average loss do not change

# TODO: make loss function modular 

(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
function (c::Chain)(x, y)
    loss = c.loss(c(x), y)
    if training() # Only apply regularization during training, only to weights, not biases.
        c.lambda1 != 0 && (loss += c.lambda1 * sum(sum(abs, l.w) for l in c.layers))
        c.lambda2 != 0 && (loss += c.lambda2 * sum(sum(abs2, l.w) for l in c.layers))
    end
    return loss
end
(c::Chain)(d::Data) = mean(c(x, y) for (x, y) in d)

# This became redundant now
struct ResLayerConv; w; padding; stride; end
# Random init
ResLayerConv(w1, w2, nx, ny; padding=0, stride=1) = ResLayerConv(param(w1, w2, nx, ny), padding, stride)
# Predeterminde weights
ResLayerConv(w; padding=0, stride=1) = ResLayerConv(param(w), padding, stride)
(rl0::ResLayerConv)(x) = conv4(rl0.w, x; padding=rl0.padding, stride=rl0.stride)

# X0
struct ResLayerX0; batch_w; conv_w; ms; padding; stride; end
# Predetermined weights
# TODO: not sure if we should make batch_w as param here + throws error
ResLayerX0(w, ms; padding=0, stride=1) = ResLayerX0(
        w[2:3],
        param(w[1]; atype=Knet.atype()), 
        ms,
        padding, 
        stride)
(rlx0::ResLayerX0)(x) =  batchnorm_as_function(rlx0.batch_w, conv4(rlx0.conv_w, x; padding=rlx0.padding, stride=rlx0.stride), rlx0.ms) 

# X1
struct ResLayerX1; x0_layer; is_initial::Bool; end
ResLayerX1(w, ms; padding=0, stride=1, is_initial::Bool=false) = ResLayerX1(ResLayerX0(w, ms; padding=padding, stride=stride), is_initial)
function (rlx1::ResLayerX1)(x)
    relu_res = relu.(rlx1.x0_layer(x))
    if rlx1.is_initial
        return pool(relu_res; window=3, stride=2)
    else
        return relu_res
    end
end


# X2
# TODO: can be constructed like Chain
struct ResLayerX2; x1_a_layer; x1_b_layer; x0_c_layer; end
ResLayerX2(w, ms; pads=[0, 1, 0], strides=[1, 1, 1]) = ResLayerX2(
    ResLayerX1(w[1:3], ms; padding=pads[1], stride=strides[1]),
    ResLayerX1(w[4:6], ms; padding=pads[2], stride=strides[2]),
    ResLayerX0(w[7:9], ms; padding=pads[3], stride=strides[3]),
    )
(rlx2::ResLayerX2)(x) =   rlx2.x0_c_layer(rlx2.x1_b_layer((rlx2.x1_a_layer(x))))


# X3
struct ResLayerX3; x0_a_layer::ResLayerX0; x2_b_layer::ResLayerX2; end
ResLayerX3(w, ms; pads=[0, 0, 1, 0], strides=[2, 2, 1, 1]) = ResLayerX3(
    ResLayerX0(w[1:3], ms; padding=pads[1], stride=strides[1]),
    ResLayerX2(w[4:12], ms; pads=pads[2:4], strides=strides[2:4])
    )
(rlx3::ResLayerX3)(x) =   relu.(rlx3.x0_a_layer(x) .+ rlx3.x2_b_layer(x))


# X4
struct ResLayerX4;  x2_layer; end
ResLayerX4(w, ms; pads=[ 0, 1, 0], strides=[1, 1, 1]) = ResLayerX4(
    ResLayerX2(w, ms; pads=pads, strides=strides)
    )
(rlx4::ResLayerX4)(x) =   relu.(x .+ rlx4.x2_layer(x))


# X5
struct ResLayerX5;
    x3_layer::ResLayerX3; x4_layers;  is_next_fc::Bool;

end

function ResLayerX5(w, ms; strides=[2, 2, 1, 1], is_next_fc::Bool=false)
    x3_layer::ResLayerX3 = ResLayerX3(w[1:12], ms; strides=strides)
    x4_layers = []
    for k = 13:9:length(w)
        layer = ResLayerX4(w[k:k + 8], ms)
        push!(x4_layers, layer)
    end
    return ResLayerX5(x3_layer, x4_layers, is_next_fc)
end 

function (rlx5::ResLayerX5)(x) 
    x = rlx5.x3_layer(x)
    for l in rlx5.x4_layers
        x = l(x)
    end

    if rlx5.is_next_fc
        return pool(x; stride=1, window=7, mode=2)
    else
        return x
    end
end









