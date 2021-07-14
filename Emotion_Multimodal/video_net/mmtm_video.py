#MMTM: #https://github.com/haamoon/mmtm/blob/master/mmtm.py
#SE Keras: https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se.py

from keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply, add, Conv3D
from keras.layers import Concatenate, Conv2D, GlobalAveragePooling2D

def squeeze_excite_block_2D3D(input_tensor_video, input_tensor_audio, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    #init = input_tensor_video # or input_tensor_audio
    #filters = input_tensor_video.shape[4] #or input_tensor_audio[3]
    filters_3D= input_tensor_video._keras_shape[-1]
    #print("The filter 3D shape:", filters_3D)
    se_shape_3D = (1, 1, 1, filters_3D)
    
    filters_2D= input_tensor_audio._keras_shape[-1]
    #print("The filter 2D shape:", filters_2D)
    se_shape_2D = (1, 1, filters_2D)

    se_GAP3D = GlobalAveragePooling3D()(input_tensor_video)
    se_GAP2D = GlobalAveragePooling2D()(input_tensor_audio)
    
    filters = filters_3D
    #print("The filter sum:", filters)
    # concatenate them
    #merged = Concatenate(axis=1)([se_GAP3D, se_GAP2D])  #tf.keras.layers.Concatenate(axis=1)([x1, x2])
    merged = add([se_GAP3D, se_GAP2D])
    #print("The concadinate shape:", merged.shape)
    
    #3D operations
    se_3D = Reshape(se_shape_3D)(merged)
    se_3D = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_3D)
    se_3D = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_3D)
    
    
    #2D operations
    se_2D = Reshape(se_shape_2D)(merged)
    se_2D = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_2D)
    se_2D = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_2D)
    
    
    vid_se = multiply([input_tensor_video, se_3D])
    aud_se = multiply([input_tensor_audio, se_2D])
    
    return vid_se, aud_se

def spatial_squeeze_excite_block_2D3D(input_tensor_video, input_tensor_audio):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se_vid = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor_video)
    spt_vid = multiply([input_tensor_video, se_vid])
    
    se_aud = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor_audio)
    spt_aud = multiply([input_tensor_audio, se_aud])
    
    return spt_vid, spt_aud


def channel_spatial_squeeze_excite_2D3D(input_tensor_video, input_tensor_audio, ratio=8):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    vid_se, aud_se = squeeze_excite_block_2D3D(input_tensor_video, input_tensor_audio, ratio)
    spt_vid, spt_aud = spatial_squeeze_excite_block_2D3D(input_tensor_video, input_tensor_audio,)

    vid_se_out = add([vid_se, spt_vid])
    aud_se_out = add([aud_se, spt_aud])
    
    return vid_se_out, aud_se_out




#**********************Only for 2D *****************************************#

def squeeze_excite_block_2D(input_tensor1, input_tensor2, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    filters_2D= input_tensor1._keras_shape[-1]
    #print("The filter 2D shape:", filters_2D)
    se_shape_2D = (1, 1, filters_2D)

    se_GAP2D1 = GlobalAveragePooling2D()(input_tensor1)
    se_GAP2D2 = GlobalAveragePooling2D()(input_tensor2)
    
    filters = filters_2D
    #print("The filter sum:", filters)
    # concatenate them
    #merged = Concatenate(axis=1)([se_GAP3D, se_GAP2D])  #tf.keras.layers.Concatenate(axis=1)([x1, x2])
    merged = add([se_GAP2D1, se_GAP2D2])
    #print("The concadinate shape:", merged.shape)
    
    #3D operations
    se_2D1 = Reshape(se_shape_2D)(merged)
    se_2D1 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_2D1)
    se_2D1 = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_2D1)
    
    
    #2D operations
    se_2D2 = Reshape(se_shape_2D)(merged)
    se_2D2 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_2D2)
    se_2D2 = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_2D2)
    
    
    t1_se = multiply([input_tensor1, se_2D1])
    t2_se = multiply([input_tensor2, se_2D2])
    
    return t1_se, t2_se


def spatial_squeeze_excite_block_2D(input_tensor1, input_tensor2):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se_t1 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor1)
    spt_t1 = multiply([input_tensor1, se_t1])
    
    se_t2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor2)
    spt_t2 = multiply([input_tensor2, se_t2])
    
    return spt_t1, spt_t2


def channel_spatial_squeeze_excite_2D(input_tensor1, input_tensor2, ratio=8):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    t1_se, t2_se = squeeze_excite_block_2D(input_tensor1, input_tensor2, ratio)
    spt_t1, spt_t2 = spatial_squeeze_excite_block_2D(input_tensor1, input_tensor2)

    vid_se_out = add([t1_se, spt_t1])
    aud_se_out = add([t2_se, spt_t2])
    
    return vid_se_out, aud_se_out



#**********************Only for 3D *****************************************#

def squeeze_excite_block_3D(input_tensor1, input_tensor2, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    filters_3D= input_tensor1._keras_shape[-1]
    #print("The filter 2D shape:", filters_2D)
    se_shape_3D = (1, 1, 1, filters_3D)

    se_GAP2D1 = GlobalAveragePooling3D()(input_tensor1)
    se_GAP2D2 = GlobalAveragePooling3D()(input_tensor2)
    
    filters = filters_3D
    #print("The filter sum:", filters)
    # concatenate them
    #merged = Concatenate(axis=1)([se_GAP3D, se_GAP2D])  #tf.keras.layers.Concatenate(axis=1)([x1, x2])
    merged = add([se_GAP2D1, se_GAP2D2])
    #print("The concadinate shape:", merged.shape)
    
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
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se_t1 = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor1)
    spt_t1 = multiply([input_tensor1, se_t1])
    
    se_t2 = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input_tensor2)
    spt_t2 = multiply([input_tensor2, se_t2])
    
    return spt_t1, spt_t2


def channel_spatial_squeeze_excite_3D(input_tensor1, input_tensor2, ratio=8):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    t1_se, t2_se = squeeze_excite_block_3D(input_tensor1, input_tensor2, ratio)
    spt_t1, spt_t2 = spatial_squeeze_excite_block_3D(input_tensor1, input_tensor2)

    vid_se_out = add([t1_se, spt_t1])
    aud_se_out = add([t2_se, spt_t2])
    
    return vid_se_out, aud_se_out

