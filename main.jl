using Revise
using Knet
using IterTools
include("data.reader.jl")
using Base.Iterators: flatten
import .Iterators: cycle, Cycle, take
using Statistics
include("utils.jl")
using Plots; default(fmt=:png, ls=:auto)
include("models.jl")
include("training.jl")
include("plots.jl")