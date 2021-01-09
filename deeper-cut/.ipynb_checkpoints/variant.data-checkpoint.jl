import Base: iterate, eltype, length, rand, repeat, summary, show
using Base.Iterators: Cycle
using Base: @propagate_inbounds, tail
using Random: randperm
using Knet: atype

mutable struct VariantData{T}
    x::Any
    y::Any
    length::Any
    imax::Any
    indices::Any
    shuffle::Any
    xsize::Any
    ysize::Any
    xtype::Any
    ytype::Any
end

function variant_minibatch(
    x,
    y;
    shuffle = false,
    xsize = size(x),
    ysize = size(y),
    xtype = (eltype(x) <: AbstractFloat ? atype() : (typeof(x).name.wrapper){eltype(x)}),
    ytype = (eltype(y) <: AbstractFloat ? atype() : (typeof(y).name.wrapper){eltype(y)}),
)
    nx = size(x)[end]
    if nx != size(y)[end]
        throw(DimensionMismatch())
    end

    imax = nx
    VariantData{Tuple{xtype,ytype}}(
        x,
        y,
        nx,
        imax,
        1:nx,
        shuffle,
        xsize,
        ysize,
        xtype,
        ytype,
    )
end

@propagate_inbounds function iterate(d::VariantData, i = 0)
    if i >= d.imax
        return nothing
    end
    if d.shuffle && i == 0
        d.indices = randperm(d.length)
    end
    nexti = min(i + 1, d.length)
    id = nexti
    xbatch =  convert(d.xtype, d.x[id])
    if d.y == nothing
        return (xbatch, nexti)
    else
        ybatch = convert(d.ytype, d.y[id])
        return ((xbatch, ybatch), nexti)
    end
end

eltype(::Type{VariantData{T}}) where {T} = T

function length(d::VariantData)
    n = d.length
    floor(Int, n)
end

function rand(d::VariantData)
    i = rand(0:(d.length-1))
    return iterate(d, i)[1]
end

# Give length info in summary:
summary(d::VariantData) = "$(length(d))-element $(typeof(d))"
show(io::IO, d::VariantData) = print(IOContext(io,:compact=>true), summary(d))
show(io::IO, ::MIME"text/plain", d::VariantData) = show(io, d)
