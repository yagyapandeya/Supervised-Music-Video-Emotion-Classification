from keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply, add, Permute, Conv3D

def squeeze_excite_block(input_tensor, ratio=8):
    #init = input_tensor
    #channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input_tensor._keras_shape[-1]   #_tensor_shape(init)[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([input_tensor, se])
    return x


def spatial_squeeze_excite_block(input_tensor):
    se = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=8):
    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x