using Knet: Knet, AutoGrad, dir, Data, minibatch, Param, @diff, value, params, grad, progress, progress!, KnetArray, load, save
using IterTools
using Base.Iterators
using Knet: Ops20.zeroone
using Printf, Random, Test, Statistics
using Plots; default(fmt=:png)

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
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) 

# Define a chain of layers and a loss function:
struct Chain
    layers; lambda1; lambda2; loss;
    Chain(layers...; lambda1=0, lambda2=0, loss=nll) = new(layers, lambda1, lambda2, loss)
end

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