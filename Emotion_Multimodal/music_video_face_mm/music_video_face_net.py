from keras import optimizers
from keras.layers import Input, MaxPooling3D, Conv3DTranspose, Conv3D, UpSampling3D, GlobalAveragePooling3D
from keras.layers import MaxPooling2D, Conv2DTranspose, Conv2D, UpSampling2D, GlobalAveragePooling2D
from keras.layers import  BatchNormalization, ReLU, ReLU, Reshape, Activation, Concatenate, Dense
from keras.layers.merge import concatenate, add
from keras.models import Model

from mmtm_video import channel_spatial_squeeze_excite_3D
from SE_Block import channel_spatial_squeeze_excite

def video_net(video_shape_slow, video_shape_fast, audio_shape, video_shape_face):
    video_input_face = Input(shape=video_shape_face)
    video_input_slow = Input(shape=video_shape_slow)
    video_input_fast = Input(shape=video_shape_fast)
    audio_input = Input(shape=audio_shape)
    
    #******************************************** START SLOW BRANCH ********************************************************#
    # BLOCK 1: Video SLOW
    conv_x1v = Conv3D(filters=16, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(video_input_slow)
    conc1xv = Concatenate(axis=4)([video_input_slow, conv_x1v])
    #bn_x1v = BatchNormalization()(conc1xv)
    
    conv_y1v = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc1xv)
    conc1yv = Concatenate(axis=4)([video_input_slow, conv_y1v])
    #bn_y1v = BatchNormalization()(conc1yv)
    
    conv_z1v = Conv3D(filters=16, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc1yv)
    conc1zv = Concatenate(axis=4)([video_input_slow, conv_z1v])
    #bn_z1v = BatchNormalization()(conc1zv)
    mp_z1v = MaxPooling3D(pool_size=(2,2, 2), padding='same')(conc1zv)
    
    st_y1v = Conv3D(filters=16, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input_slow)
    conc1stv = Concatenate(axis=4)([video_input_slow, st_y1v]) #Concatenate([video_input, st_y1v], axis=4)
    #bn_st1v = BatchNormalization()(conc1stv)
    mp_st1v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc1stv)
    
    out_1v = add([mp_st1v, mp_z1v])
    out_c1v = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_1v)
    out_c1v = BatchNormalization()(out_c1v)
    
    # BLOCK 1: Video FAST
    conv_x1vf = Conv3D(filters=8, kernel_size=(1, 9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(video_input_fast)
    conc1xvf = Concatenate(axis=4)([video_input_fast, conv_x1vf])
    #bn_x1vf = BatchNormalization()(conc1xvf)
    
    conv_y1vf = Conv3D(filters=1, kernel_size=(9, 9, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc1xvf)
    conc1yvf = Concatenate(axis=4)([video_input_fast, conv_y1vf])
    #bn_y1vf = BatchNormalization()(conc1yvf)
    
    conv_z1vf = Conv3D(filters=16, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(conc1yvf)
    conc1zvf = Concatenate(axis=4)([video_input_fast, conv_z1vf])
    #bn_z1vf = BatchNormalization()(conc1zvf)
    mp_z1vf = MaxPooling3D(pool_size=(2,2, 2), padding='same')(conc1zvf)
    
    st_y1vf = Conv3D(filters=16, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input_fast)
    conc1stvf = Concatenate(axis=4)([video_input_fast, st_y1vf]) #Concatenate([video_input, st_y1v], axis=4)
    #bn_st1vf = BatchNormalization()(conc1stvf)
    mp_st1vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc1stvf)
    
    out_1vf = add([mp_st1vf, mp_z1vf])
    out_c1vf = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_1vf)
    out_c1vf = BatchNormalization()(out_c1vf)
    
    # Block 1: MMIM
    out_c1vx, out_c1vxf = channel_spatial_squeeze_excite_3D(out_c1v, out_c1vf, ratio=4)
    
    # BLOCK 2:C3D SLOW
    conv_x2v = Conv3D(filters=32, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc2xv = Concatenate(axis=4)([out_c1vx, conv_x2v])
    
    conv_y2v = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc2xv)
    conc2yv = Concatenate(axis=4)([out_c1vx, conv_y2v])
    
    conv_z2v = Conv3D(filters=32, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc2yv)
    conc2zv = Concatenate(axis=4)([out_c1vx, conv_z2v])
    mp_z2v = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conc2zv)
    
    st_y2v = Conv3D(filters=32, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc2stv = Concatenate(axis=4)([out_c1vx, st_y2v])
    mp_st2v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc2stv)
    
    out_2v = add([mp_st2v, mp_z2v])
    out_c2v = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_2v)
    out_c2v = BatchNormalization()(out_c2v)
    
    # BLOCK 2:C3D FAST
    conv_x2vf = Conv3D(filters=16, kernel_size=(1, 7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vxf)
    conc2xvf = Concatenate(axis=4)([out_c1vxf, conv_x2vf])
    
    conv_y2vf = Conv3D(filters=1, kernel_size=(7, 7, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc2xvf)
    conc2yvf = Concatenate(axis=4)([out_c1vxf, conv_y2vf])
    
    conv_z2vf = Conv3D(filters=32, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(conc2yvf)
    conc2zvf = Concatenate(axis=4)([out_c1vxf, conv_z2vf])
    mp_z2vf = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conc2zvf)
    
    st_y2vf = Conv3D(filters=32, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vxf)
    conc2stvf = Concatenate(axis=4)([out_c1vxf, st_y2vf])
    mp_st2vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc2stvf)
    
    out_2vf = add([mp_st2vf, mp_z2vf])
    out_c2vf = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_2vf)
    out_c2vf = BatchNormalization()(out_c2vf)
    
    # Block 2: MMTM
    out_c2vx, out_c2vxf = channel_spatial_squeeze_excite_3D(out_c2v, out_c2vf, ratio=8)
     
    # BLOCK 3: C3D SLOW
    conv_x3v = Conv3D(filters=64, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc3xv = Concatenate(axis=4)([out_c2vx, conv_x3v])
    
    conv_y3v = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc3xv)
    conc3yv = Concatenate(axis=4)([out_c2vx, conv_y3v])
    
    conv_z3v = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc3yv)
    conc3zv = Concatenate(axis=4)([out_c2vx, conv_z3v])
    mp_z3v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc3zv)
    
    st_y3v = Conv3D(filters=64, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc3stv = Concatenate(axis=4)([out_c2vx, st_y3v])
    mp_st3v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc3stv)
    
    out_3v = add([mp_st3v, mp_z3v])
    out_c3v = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_3v)
    out_c3v = BatchNormalization()(out_c3v)
    
    # BLOCK 3: C3D FAST
    conv_x3vf = Conv3D(filters=32, kernel_size=(1, 7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vxf)
    conc3xvf = Concatenate(axis=4)([out_c2vxf, conv_x3vf])
    
    conv_y3vf = Conv3D(filters=1, kernel_size=(7, 7, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc3xvf)
    conc3yvf = Concatenate(axis=4)([out_c2vxf, conv_y3vf])
    
    conv_z3vf = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc3yvf)
    conc3zvf = Concatenate(axis=4)([out_c2vxf, conv_z3vf])
    mp_z3vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc3zvf)
    
    st_y3vf = Conv3D(filters=64, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vxf)
    conc3stvf = Concatenate(axis=4)([out_c2vxf, st_y3vf])
    mp_st3vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc3stvf)
    
    out_3vf = add([mp_st3vf, mp_z3vf])
    out_c3vf = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_3vf)
    out_c3vf = BatchNormalization()(out_c3vf)
    
    # Block 3: MMTM
    out_c3vx, out_c3vxf = channel_spatial_squeeze_excite_3D(out_c3v, out_c3vf, ratio=16)
    
    # BLOCK 4: C3D SLOW
    conv_x4v = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vx)
    conc4xv = Concatenate(axis=4)([out_c3vx, conv_x4v])
    
    conv_y4v = Conv3D(filters=1, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc4xv)
    conc4yv = Concatenate(axis=4)([out_c3vx, conv_y4v])
    
    conv_z4v = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc4yv)
    conc4zv = Concatenate(axis=4)([out_c3vx, conv_z4v])
    mp_z4v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc4zv)
    
    st_y4v = Conv3D(filters=128, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vx)
    conc4stv = Concatenate(axis=4)([out_c3vx, st_y4v])
    mp_st4v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc4stv)
    
    out_4v = add([mp_st4v, mp_z4v])
    out_c4v = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_4v)
    out_c4v = BatchNormalization()(out_c4v)
    
    # BLOCK 4: C3D FAST
    conv_x4vf = Conv3D(filters=64, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vxf)
    conc4xvf = Concatenate(axis=4)([out_c3vxf, conv_x4vf])
    
    conv_y4vf = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc4xvf)
    conc4yvf = Concatenate(axis=4)([out_c3vxf, conv_y4vf])
    
    conv_z4vf = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc4yvf)
    conc4zvf = Concatenate(axis=4)([out_c3vxf, conv_z4vf])
    mp_z4vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc4zvf)
    
    st_y4vf = Conv3D(filters=128, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vxf)
    conc4stvf = Concatenate(axis=4)([out_c3vxf, st_y4vf])
    mp_st4vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc4stvf)
    
    out_4vf = add([mp_st4vf, mp_z4vf])
    out_c4vf = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_4vf)
    out_c4vf = BatchNormalization()(out_c4vf)
    
    # Block 4: MMTM
    out_c4vx, out_c4vxf = channel_spatial_squeeze_excite_3D(out_c4v, out_c4vf, ratio=16)
    
    # BLOCK 5: C3D SLOW
    conv_x5v = Conv3D(filters=256, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4vx)
    conc5xv = Concatenate(axis=4)([out_c4vx, conv_x5v])
    
    conv_y5v = Conv3D(filters=1, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc5xv)
    conc5yv = Concatenate(axis=4)([out_c4vx, conv_y5v])
    
    conv_z5v = Conv3D(filters=256, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc5yv)
    conc5zv = Concatenate(axis=4)([out_c4vx, conv_z5v])
    mp_z5v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc5zv)
    
    st_y5v = Conv3D(filters=256, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4vx)
    conc5stv = Concatenate(axis=4)([out_c4vx, st_y5v])
    mp_st5v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc5stv)
    
    out_5v = add([mp_st5v, mp_z5v])
    out_c5v = Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_5v)
    out_c5v = BatchNormalization()(out_c5v)
    
    # BLOCK 5: C3D FAST
    conv_x5vf = Conv3D(filters=128, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4vxf)
    conc5xvf = Concatenate(axis=4)([out_c4vxf, conv_x5vf])
    
    conv_y5vf = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc5xvf)
    conc5yvf = Concatenate(axis=4)([out_c4vxf, conv_y5vf])
    
    conv_z5vf = Conv3D(filters=256, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc5yvf)
    conc5zvf = Concatenate(axis=4)([out_c4vxf, conv_z5vf])
    mp_z5vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc5zvf)
    
    st_y5vf = Conv3D(filters=256, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4vxf)
    conc5stvf = Concatenate(axis=4)([out_c4vxf, st_y5vf])
    mp_st5vf = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc5stvf)
    
    out_5vf = add([mp_st5vf, mp_z5vf])
    out_c5vf = Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_5vf)
    out_c5vf = BatchNormalization()(out_c5vf)
    
    # Block 4: MMTM
    out_c5vx, out_c5vxf = channel_spatial_squeeze_excite_3D(out_c5v, out_c5vf, ratio=16)
    
    
    #******************************MUSIC 2D Network **********************************************************#
     # BLOCK 1: AUDIO
    conv_x1a = Conv2D(filters=8, kernel_size=(13, 13), padding='same', activation='relu', kernel_initializer='he_normal')(audio_input)
    conc1xa = Concatenate(axis=3)([audio_input, conv_x1a])
    bn_x1a = BatchNormalization()(conc1xa)
    
    conv_y1a = Conv2D(filters=8, kernel_size=(9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x1a)
    conc1ya = Concatenate(axis=3)([audio_input, conv_y1a])
    bn_y1a = BatchNormalization()(conc1ya)
    
    conv_z1a = Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y1a)
    conc1za = Concatenate(axis=3)([audio_input, conv_z1a])
    bn_z1a= BatchNormalization()(conc1za)
    mp_z1a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_z1a)
    
    st_y1a = Conv2D(filters=8, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(audio_input)
    conc1sta = Concatenate(axis=3)([audio_input, st_y1a])
    bn_st1a = BatchNormalization()(conc1sta)
    mp_st1a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_st1a)
    
    out_1a = add([mp_st1a, mp_z1a])
    out_c1a = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_1a)
    out_c1ax = BatchNormalization()(out_c1a)
    
    # BLOCK 2: AUDIO
    conv_x2a = Conv2D(filters=16, kernel_size=(9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1ax)
    conc2xa = Concatenate(axis=3)([out_c1ax, conv_x2a])
    bn_x2a = BatchNormalization()(conc2xa)
    
    conv_y2a = Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x2a)
    conc2ya = Concatenate(axis=3)([out_c1ax, conv_y2a])
    bn_y2a = BatchNormalization()(conc2ya)
    
    conv_z2a = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y2a)
    conc2za = Concatenate(axis=3)([out_c1ax, conv_z2a])
    bn_z2a = BatchNormalization()(conc2za)
    mp_z2a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_z2a)
    
    st_y2a = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu',kernel_initializer='he_normal')(out_c1ax)
    conc2sta = Concatenate(axis=3)([out_c1ax, st_y2a])
    bn_st2a = BatchNormalization()(conc2sta)
    mp_st2a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_st2a)
    
    out_2a = add([mp_st2a, mp_z2a])
    out_c2a = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_2a)
    out_c2ax = BatchNormalization()(out_c2a)
    
    # BLOCK 3: AUDIO
    conv_x3a = Conv2D(filters=32, kernel_size=(9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2ax)
    conc3xa = Concatenate(axis=3)([out_c2ax, conv_x3a])
    bn_x3a = BatchNormalization()(conc3xa)
    
    conv_y3a = Conv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x3a)
    conc3ya = Concatenate(axis=3)([out_c2ax, conv_y3a])
    bn_y3a = BatchNormalization()(conc3ya)
    
    conv_z3a = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y3a)
    conc3za = Concatenate(axis=3)([out_c2ax, conv_z3a])
    bn_z3a = BatchNormalization()(conc3za)
    mp_z3a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_z3a)
    
    st_y3a = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2ax)
    conc3sta = Concatenate(axis=3)([out_c2ax, st_y3a])
    bn_st3a = BatchNormalization()(conc3sta)
    mp_st3a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_st3a)
    
    out_3a = add([mp_st3a, mp_z3a])
    out_c3a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_3a)
    out_c3ax = BatchNormalization()(out_c3a)
    
    # BLOCK 4: AUDIO
    conv_x4a = Conv2D(filters=64, kernel_size=(9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3ax)
    conc4xa = Concatenate(axis=3)([out_c3ax, conv_x4a])
    bn_x4a = BatchNormalization()(conc4xa)
    
    conv_y4a = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x4a)
    conc4ya = Concatenate(axis=3)([out_c3ax, conv_y4a])
    bn_y4a = BatchNormalization()(conc4ya)
    
    conv_z4a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y4a)
    conc4za = Concatenate(axis=3)([out_c3ax, conv_z4a])
    bn_z4a = BatchNormalization()(conc4za)
    mp_z4a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_z4a)
    
    st_y4a = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3ax)
    conc4sta = Concatenate(axis=3)([out_c3ax, st_y4a])
    bn_st4a = BatchNormalization()(conc4sta)
    mp_st4a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_st4a)
    
    out_4a = add([mp_st4a, mp_z4a])
    out_c4a = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_4a)
    out_c4ax = BatchNormalization()(out_c4a)
    
    # BLOCK 5: AUDIO
    conv_x5a = Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4ax)
    conc5xa = Concatenate(axis=3)([out_c4ax, conv_x5a])
    bn_x5a = BatchNormalization()(conc5xa)
    #bn_x5a = ReLU()(bn_x5a)
    
    conv_y5a = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x5a)
    conc5ya = Concatenate(axis=3)([out_c4ax, conv_y5a])
    bn_y5a = BatchNormalization()(conc5ya)
    #bn_y5a = ReLU()(bn_y5a)
    
    conv_z5a = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y5a)
    conc5za = Concatenate(axis=3)([out_c4ax, conv_z5a])
    bn_z5a = BatchNormalization()(conc5za)
    #bn_z5a = ReLU()(bn_z5a)
    mp_z5a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_z5a)
    
    # expand channels for the sum
    st_y5a = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4ax)
    conc5sta = Concatenate(axis=3)([out_c4ax, st_y5a])
    bn_st5a = BatchNormalization()(conc5sta)
    #bn_st5a = ReLU()(bn_st5a)
    mp_st5a = MaxPooling2D(pool_size=(2,2), padding='same')(bn_st5a)
    
    out_5a = add([mp_st5a, mp_z5a])
    out_c5a = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_5a)
    out_c5ax = BatchNormalization()(out_c5a)
    #out_c5a = ReLU()(out_c5a)
    
    #********************************************* END MUSIC NETWORK ******************************************************#
    
    
    #********************************************* SART FACE FAST NETWORK ******************************************************#

    # BLOCK 1: Video Slow
    conv_fcx1 = Conv3D(filters=4, kernel_size=(1, 9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(video_input_face)
    conc_fcx1 = Concatenate(axis=4)([video_input_face, conv_fcx1])
    #bn_x1vf = BatchNormalization()(conc1xvf)
    
    conv_fcy1 = Conv3D(filters=1, kernel_size=(9, 9, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcx1)
    conc_fcy1 = Concatenate(axis=4)([video_input_face, conv_fcy1])
    #bn_y1vf = BatchNormalization()(conc1yvf)
    
    conv_fcz1 = Conv3D(filters=4, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcy1)
    conc_fcz1 = Concatenate(axis=4)([video_input_face, conv_fcz1])
    #bn_z1vf = BatchNormalization()(conc1zvf)
    mp_fcz1 = MaxPooling3D(pool_size=(2,2, 2), padding='same')(conc_fcz1)
    
    conv_fcs1 = Conv3D(filters=4, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input_face)
    conc_fcs1 = Concatenate(axis=4)([video_input_face, conv_fcs1]) #Concatenate([video_input, st_y1v], axis=4)
    #bn_st1vf = BatchNormalization()(conc1stvf)
    mp_fcs1 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcs1)
    
    out_add1 = add([mp_fcs1, mp_fcz1])
    conv_fco1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_add1)
    bn_fco1 = BatchNormalization()(conv_fco1)
    
    se_fco1 = channel_spatial_squeeze_excite(bn_fco1, 4)
    
    # BLOCK 2:C3D
    conv_fcx2 = Conv3D(filters=8, kernel_size=(1, 7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco1)
    conc_fcx2 = Concatenate(axis=4)([se_fco1, conv_fcx2])
    #bn_x2vf = BatchNormalization()(conc2xvf)
    
    conv_fcy2 = Conv3D(filters=1, kernel_size=(7, 7, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcx2)
    conc_fcy2 = Concatenate(axis=4)([se_fco1, conv_fcy2])
    #bn_y2vf = BatchNormalization()(conc2yvf)
    
    conv_fcz2 = Conv3D(filters=8, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcy2)
    conc_fcz2 = Concatenate(axis=4)([se_fco1, conv_fcz2])
    #bn_z2vf = BatchNormalization()(conc2zvf)
    mp_fcz2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conc_fcz2)
    
    conv_fcs2 = Conv3D(filters=8, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco1)
    conc_fcs2 = Concatenate(axis=4)([se_fco1, conv_fcs2])
    #bn_st2vf = BatchNormalization()(conc2stvf)
    mp_fcs2 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcs2)
    
    out_add2 = add([mp_fcs2, mp_fcz2])
    conv_fco2 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_add2)
    bn_fco2 = BatchNormalization()(conv_fco2)
    
    se_fco2 = channel_spatial_squeeze_excite(bn_fco2, 8)
       
    # BLOCK 3: C3D
    conv_fcx3 = Conv3D(filters=16, kernel_size=(1, 7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco2)
    conc_fcx3 = Concatenate(axis=4)([se_fco2, conv_fcx3])
    #bn_x3vf = BatchNormalization()(conc3xvf)
    
    conv_fcy3 = Conv3D(filters=1, kernel_size=(7, 7, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcx3)
    conc_fcy3 = Concatenate(axis=4)([se_fco2, conv_fcy3])
    #bn_y3vf = BatchNormalization()(conc3yvf)
    
    conv_fcz3 = Conv3D(filters=16, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcy3)
    conc_fcz3 = Concatenate(axis=4)([se_fco2, conv_fcz3])
    #bn_z3vf = BatchNormalization()(conc3zvf)
    mp_fcz3 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcz3)
    
    conv_fcs3 = Conv3D(filters=16, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco2)
    conc_fcs3 = Concatenate(axis=4)([se_fco2, conv_fcs3])
    #bn_st3vf = BatchNormalization()(conc3stvf)
    mp_fcs3 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcs3)
    
    out_add3 = add([mp_fcs3, mp_fcz3])
    conv_fco3 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_add3)
    bn_fco3 = BatchNormalization()(conv_fco3)
    
    se_fco3 = channel_spatial_squeeze_excite(bn_fco3, 8)
    
    # BLOCK 4: C3D
    conv_fcx4 = Conv3D(filters=32, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco3)
    conc_fcx4 = Concatenate(axis=4)([se_fco3, conv_fcx4])
    #bn_x4vf = BatchNormalization()(conc4xvf)
    
    conv_fcy4 = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcx4)
    conc_fcy4 = Concatenate(axis=4)([se_fco3, conv_fcy4])
    #bn_y4vf = BatchNormalization()(conc4yvf)
    
    conv_fcz4 = Conv3D(filters=32, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcy4)
    conc_fcz4 = Concatenate(axis=4)([se_fco3, conv_fcz4])
    #bn_z4vf = BatchNormalization()(conc4zvf)
    mp_fcz4 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcz4)
    
    # expand channels for the sum
    conv_fcs4 = Conv3D(filters=32, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco3)
    conc_fcs4 = Concatenate(axis=4)([se_fco3, conv_fcs4])
    #bn_st4vf = BatchNormalization()(conc4stvf)
    mp_fcs4 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcs4)
    
    out_add4 = add([mp_fcs4, mp_fcz4])
    conv_fco4 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_add4)
    bn_fco4 = BatchNormalization()(conv_fco4)
    
    se_fco4 = channel_spatial_squeeze_excite(bn_fco4, 16)
        
    # BLOCK 5: C3D
    conv_fcx5 = Conv3D(filters=64, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco4)
    conc_fcx5 = Concatenate(axis=4)([se_fco4, conv_fcx5])
    #bn_x5vf = BatchNormalization()(conc5xvf)
    
    conv_fcy5 = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcx5)
    conc_fcy5 = Concatenate(axis=4)([se_fco4, conv_fcy5])
    #bn_y5vf = BatchNormalization()(conc5yvf)
    
    conv_fcz5 = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc_fcy5)
    conc_fcz5 = Concatenate(axis=4)([se_fco4, conv_fcz5])
    #bn_z5vf = BatchNormalization()(conc5zvf)
    mp_fcz5 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcz5)
    
    conv_fcs5 = Conv3D(filters=64, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(se_fco4)
    conc_fcs5 = Concatenate(axis=4)([se_fco4, conv_fcs5])
    #bn_st5vf = BatchNormalization()(conc5stvf)
    mp_fcs5 = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc_fcs5)
    
    out_add5 = add([mp_fcs5, mp_fcz5])
    conv_fco5 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_add5)
    bn_fco5 = BatchNormalization()(conv_fco5)
    
    se_fco5 = channel_spatial_squeeze_excite(bn_fco5, 16)
    
    gap_3D_face = GlobalAveragePooling3D()(se_fco5)
    #print("The size of gap_3D:", np.shape(gap_3D_face))
    
    #********************************************* END SART FACE FAST NETWORK ******************************************************#
    
    gap_2D_slow = GlobalAveragePooling2D()(out_c5ax)
    
    gap_3D_slow = GlobalAveragePooling3D()(out_c5vx)
    gap_3D_fast = GlobalAveragePooling3D()(out_c5vxf)
 
    final_out = concatenate([gap_3D_slow, gap_3D_fast, gap_2D_slow, gap_3D_face])
    final_out = Dense(6, activation='softmax', name='AV_OUT')(final_out)
    
    model = Model(inputs=[video_input_slow, video_input_fast, audio_input, video_input_face], outputs=[final_out])
    model.summary()
    
    model.load_weights('./music_video_face_mm_best2.h5', by_name = True)
    print("The pre-trained weight loaded.")
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-6),metrics=["accuracy"])  

    return model
    
