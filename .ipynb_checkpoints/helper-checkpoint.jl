using Knet

add_dim(x::Array) = reshape(x, (size(x)..., 1));

# Not working as expected
function concatenate_iterables(iterables...)
    return zip(iterables...) |> Iterators.flatten
end

# mutable struct Data{T}; x; y; batchsize; length; partial; imax; indices; shuffle; xsize; ysize; xtype; ytype; end
function append_to_data!(data::Knet.Train20.Data, another_data::Knet.Train20.Data)
    println(size(data.x))
data.x = cat(data.x, another_data.x, dims=2)
data.y = cat(data.y, another_data.y, dims=2)
    data.length = data.length + another_data.length
    data.imax = data.length - data.batchsize + 1
    data.indices = 1:data.length
end