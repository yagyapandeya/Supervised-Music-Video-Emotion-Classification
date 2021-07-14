#https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se.py

from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Conv2D

def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor._keras_shape[-1]  
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([input_tensor, se])
    return x

def spatial_squeeze_excite_block(input_tensor):
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor)
    x = multiply([input_tensor, se])
    return x

def channel_spatial_squeeze_excite2D(input_tensor, ratio=8):
    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x