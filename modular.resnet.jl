import CUDA
include("models.jl")
using MAT, OffsetArrays, FFTViews, ArgParse, Images, ImageMagick, Knet

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