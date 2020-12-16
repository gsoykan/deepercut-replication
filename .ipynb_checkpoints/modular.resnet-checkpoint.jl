import CUDA
include("models.jl")
using MAT, OffsetArrays, FFTViews, ArgParse, Images, ImageMagick, Knet

#=
DeeperCut Training Recipe

- Then, we decrease the stride of the first convolutional layers of the conv5 bank from 2 px 
to 1 px to prevent down-sampling. 

- Next, we add holes to all 3x3 convolutions in conv5 to preserve their receptive field. This reduces the stride of the full CNN to 16 px.

=#



function generate_resnet_from_weights(w, ms)
    conv1 = ResLayerX1(w[1:3], ms; padding=3, stride=2, is_initial=true)
    r2 = ResLayerX5(w[4:33], ms; strides=[1, 1, 1, 1])
    r3 = ResLayerX5(w[34:108], ms)
    r4 = ResLayerX5(w[109:435], ms)
    r5 = ResLayerX5(w[436:465], ms; is_next_fc=true)
    fc = Dense(w[466], w[467])
    return Chain(
        conv1,
        r2,
        r3,
        r4,
        r5,
        fc
    )
end

function generate_headless_resnet_from_weights(w, ms)
    conv1 = ResLayerX1(w[1:3], ms; padding=3, stride=2, is_initial=true)
    r2 = ResLayerX5(w[4:33], ms; strides=[1, 1, 1, 1])
    r3 = ResLayerX5(w[34:108], ms)
    r4 = ResLayerX5(w[109:435], ms)
    r5 = ResLayerX5(w[436:465], ms; strides=[1,1,1,1])
    return Chain(
        conv1,
        r2,
        r3,
        r4,
        r5
    )
end

function generate_resnet50_from_weights(w, ms)
    layer1 = ResLayerX1_50(w[1:4], ms)
    r2 = ResLayerX5(w[5:34], ms; strides=[1, 1, 1, 1])
    r3 = ResLayerX5(w[35:73], ms)
    r4 = ResLayerX5(w[74:130], ms)
    r5 = ResLayerX5(w[131:160], ms; is_next_fc=true)
    fc = Dense(w[161], w[162])
        return Chain(
        layer1,
        r2,
        r3,
        r4,
        r5,
        fc
    )
end

function generate_headless_resnet50_from_weights(w, ms)
    layer1 = ResLayerX1_50(w[1:4], ms)
    r2 = ResLayerX5(w[5:34], ms; strides=[1, 1, 1, 1])
    r3 = ResLayerX5(w[35:73], ms)
    r4 = ResLayerX5(w[74:130], ms)
    r5 = ResLayerX5(w[131:160], ms; strides=[1,1,1,1], b_layer_dilations=[1, 2, 1], b_layer_pads=[0,2,0])
        return Chain(
        layer1,
        r2,
        r3,
        r4,
        r5
    )
end

#=
# mode, 0=>train, 1=>test
function resnet50(w,x,ms; mode=1)
    # layer 1
    conv1  = conv4(w[1],x; padding=3, stride=2) .+ w[2]
    bn1    = batchnorm(w[3:4],conv1,ms; mode=mode)
    pool1  = pool(bn1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[5:34], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[35:73], r2, ms; mode=mode)
    r4 = reslayerx5(w[74:130], r3, ms; mode=mode) # 5
    r5 = reslayerx5(w[131:160], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[161] * mat(pool5) .+ w[162]
end

# mode, 0=>train, 1=>test
function resnet101(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:72], r2, ms; mode=mode)
    r4 = reslayerx5(w[73:282], r3, ms; mode=mode)
    r5 = reslayerx5(w[283:312], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[313] * mat(pool5) .+ w[314]
end

# mode, 0=>train, 1=>test
function resnet152(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:108], r2, ms; mode=mode)
    r4 = reslayerx5(w[109:435], r3, ms; mode=mode)
    r5 = reslayerx5(w[436:465], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[466] * mat(pool5) .+ w[467]
end

function reslayerx0(w,x,ms; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms; mode=mode)
end

function reslayerx1(w,x,ms; padding=0, stride=1, mode=1)
    relu.(reslayerx0(w,x,ms; padding=padding, stride=stride, mode=mode))
end
=#