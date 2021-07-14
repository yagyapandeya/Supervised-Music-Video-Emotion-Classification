from keras import optimizers
from keras.layers import Input, MaxPooling3D, Conv3DTranspose, Conv3D, UpSampling3D, GlobalAveragePooling3D
from keras.layers import MaxPooling2D, Conv2DTranspose, Conv2D, UpSampling2D, GlobalAveragePooling2D
from keras.layers import  BatchNormalization, ReLU, ReLU, Reshape, Activation, Concatenate, Dense
from keras.layers.merge import concatenate, add
from keras.models import Model

from keras.layers.convolutional import Deconv3D, Deconv2D, ZeroPadding3D

from mmtm_video import channel_spatial_squeeze_excite_3D


def video_net(video_shape_slow, video_shape_fast):

    video_input_slow = Input(shape=video_shape_slow)
    video_input_fast = Input(shape=video_shape_fast)
    
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
    
    gap_3D_slow = GlobalAveragePooling3D()(out_c5vx)
    gap_3D_fast = GlobalAveragePooling3D()(out_c5vxf)
 
    final_out = concatenate([gap_3D_slow, gap_3D_fast])
    final_out = Dense(6, activation='softmax', name='AV_OUT')(final_out)
    
    model = Model(inputs=[video_input_slow, video_input_fast], outputs=[final_out])
    model.summary()
    
    model.load_weights('./video_net_wt_best.h5', by_name = True)
    print("The pre-trained weight loaded.")
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=5e-5),metrics=["accuracy"])  

    return model
    
