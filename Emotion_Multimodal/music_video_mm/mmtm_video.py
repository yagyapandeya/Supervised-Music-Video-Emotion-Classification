from keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply, add, Conv3D
from keras.layers import Concatenate, Conv2D, GlobalAveragePooling2D

def squeeze_excite_block_3D(input_tensor1, input_tensor2, ratio=16):
    filters_3D= input_tensor1._keras_shape[-1]
    se_shape_3D = (1, 1, 1, filters_3D)

    se_GAP2D1 = GlobalAveragePooling3D()(input_tensor1)
    se_GAP2D2 = GlobalAveragePooling3D()(input_tensor2)
    
    filters = filters_3D
    merged = add([se_GAP2D1, se_GAP2D2])
    
    #3D operations
    se_2D1 = Reshape(se_shape_3D)(merged)
    se_2D1 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_2D1)
    se_2D1 = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_2D1)
    
    
    #2D operations
    se_2D2 = Reshape(se_shape_3D)(merged)
    se_2D2 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_2D2)
    se_2D2 = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_2D2)
    
    
    t1_se = multiply([input_tensor1, se_2D1])
    t2_se = multiply([input_tensor2, se_2D2])
    
    return t1_se, t2_se


def spatial_squeeze_excite_block_3D(input_tensor1, input_tensor2):
    se_t1 = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor1)
    spt_t1 = multiply([input_tensor1, se_t1])
    
    se_t2 = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor2)
    spt_t2 = multiply([input_tensor2, se_t2])
    
    return spt_t1, spt_t2


def channel_spatial_squeeze_excite_3D(input_tensor1, input_tensor2, ratio=8):
    t1_se, t2_se = squeeze_excite_block_3D(input_tensor1, input_tensor2, ratio)
    spt_t1, spt_t2 = spatial_squeeze_excite_block_3D(input_tensor1, input_tensor2)

    vid_se_out = add([t1_se, spt_t1])
    aud_se_out = add([t2_se, spt_t2])
    
    return vid_se_out, aud_se_out

