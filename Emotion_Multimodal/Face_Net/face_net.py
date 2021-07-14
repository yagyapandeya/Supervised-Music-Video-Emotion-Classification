from keras import optimizers
from keras.layers import Input, MaxPooling3D, Conv3D, GlobalAveragePooling3D, Conv3DTranspose, Conv2DTranspose
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D
from keras.layers import  BatchNormalization, ReLU, Concatenate, Dense
from keras.layers.merge import concatenate, add
from keras.models import Model

from SE_Block import channel_spatial_squeeze_excite

def video_audio_net(video_shape):
    
    video_input = Input(shape=video_shape)
    
    # BLOCK 1: C3D
    conv_x1v = Conv3D(filters=8, kernel_size=(13, 13, 13), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc1xv = Concatenate(axis=4)([video_input, conv_x1v])
    bn_x1v = BatchNormalization()(conc1xv)
    
    conv_y1v = Conv3D(filters=8, kernel_size=(9, 9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x1v)
    conc1yv = Concatenate(axis=4)([video_input, conv_y1v])
    bn_y1v = BatchNormalization()(conc1yv)
    
    conv_z1v = Conv3D(filters=8, kernel_size=(5, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y1v)
    conc1zv = Concatenate(axis=4)([video_input, conv_z1v])
    bn_z1v = BatchNormalization()(conc1zv)
    mp_z1v = MaxPooling3D(pool_size=(2,2, 2), padding='same')(bn_z1v)
    
    
    st_y1v = Conv3D(filters=8, kernel_size=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc1stv = Concatenate(axis=4)([video_input, st_y1v]) #Concatenate([video_input, st_y1v], axis=4)
    bn_st1v = BatchNormalization()(conc1stv)
    mp_st1v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_st1v)
    
    out_1v = add([mp_st1v, mp_z1v])
    out_c1v = Conv3D(filters=16, kernel_size=(5, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_1v)
    out_c1v = BatchNormalization()(out_c1v)
    
    out_c1v = channel_spatial_squeeze_excite(out_c1v, 4)
    
    # BLOCK 2:C3D
    conv_x2v = Conv3D(filters=16, kernel_size=(9, 9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1v)
    conc2xv = Concatenate(axis=4)([out_c1v, conv_x2v])
    bn_x2v = BatchNormalization()(conc2xv)
    
    conv_y2v = Conv3D(filters=16, kernel_size=(7, 7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x2v)
    conc2yv = Concatenate(axis=4)([out_c1v, conv_y2v])
    bn_y2v = BatchNormalization()(conc2yv)
    
    conv_z2v = Conv3D(filters=16, kernel_size=(5, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y2v)
    conc2zv = Concatenate(axis=4)([out_c1v, conv_z2v])
    bn_z2v = BatchNormalization()(conc2zv)
    mp_z2v = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(bn_z2v)
    
    st_y2v = Conv3D(filters=16, kernel_size=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1v)
    conc2stv = Concatenate(axis=4)([out_c1v, st_y2v])
    bn_st2v = BatchNormalization()(conc2stv)
    mp_st2v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_st2v)
    
    out_2v = add([mp_st2v, mp_z2v])
    out_c2v = Conv3D(filters=32, kernel_size=(5, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_2v)
    out_c2v = BatchNormalization()(out_c2v)
    
    out_c2v = channel_spatial_squeeze_excite(out_c2v, 4)
    
    # BLOCK 3: C3D
    conv_x3v = Conv3D(filters=32, kernel_size=(9, 9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2v)
    conc3xv = Concatenate(axis=4)([out_c2v, conv_x3v])
    bn_x3v = BatchNormalization()(conc3xv)
    
    conv_y3v = Conv3D(filters=32, kernel_size=(7, 7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x3v)
    conc3yv = Concatenate(axis=4)([out_c2v, conv_y3v])
    bn_y3v = BatchNormalization()(conc3yv)
    
    conv_z3v = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y3v)
    conc3zv = Concatenate(axis=4)([out_c2v, conv_z3v])
    bn_z3v = BatchNormalization()(conc3zv)
    mp_z3v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_z3v)
    
    st_y3v = Conv3D(filters=32, kernel_size=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2v)
    conc3stv = Concatenate(axis=4)([out_c2v, st_y3v])
    bn_st3v = BatchNormalization()(conc3stv)
    mp_st3v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_st3v)
    
    out_3v = add([mp_st3v, mp_z3v])
    out_c3v = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_3v)
    out_c3v = BatchNormalization()(out_c3v)
    
    out_c3v = channel_spatial_squeeze_excite(out_c3v, 8)
    
    # BLOCK 4: C3D
    conv_x4v = Conv3D(filters=64, kernel_size=(9, 9, 9), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3v)
    conc4xv = Concatenate(axis=4)([out_c3v, conv_x4v])
    bn_x4v = BatchNormalization()(conc4xv)
    
    conv_y4v = Conv3D(filters=64, kernel_size=(5, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x4v)
    conc4yv = Concatenate(axis=4)([out_c3v, conv_y4v])
    bn_y4v = BatchNormalization()(conc4yv)
    
    conv_z4v = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y4v)
    conc4zv = Concatenate(axis=4)([out_c3v, conv_z4v])
    bn_z4v = BatchNormalization()(conc4zv)
    mp_z4v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_z4v)
    
    st_y4v = Conv3D(filters=64, kernel_size=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3v)
    conc4stv = Concatenate(axis=4)([out_c3v, st_y4v])
    bn_st4v = BatchNormalization()(conc4stv)
    mp_st4v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_st4v)
    
    out_4v = add([mp_st4v, mp_z4v])
    out_c4v = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_4v)
    out_c4v = BatchNormalization()(out_c4v)
    
    out_c4v = channel_spatial_squeeze_excite(out_c4v, 16)
   
    # BLOCK 5: C3D
    conv_x5v = Conv3D(filters=128, kernel_size=(7, 7, 7), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4v)
    conc5xv = Concatenate(axis=4)([out_c4v, conv_x5v])
    bn_x5v = BatchNormalization()(conc5xv)
    
    conv_y5v = Conv3D(filters=128, kernel_size=(5, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(bn_x5v)
    conc5yv = Concatenate(axis=4)([out_c4v, conv_y5v])
    bn_y5v = BatchNormalization()(conc5yv)
    
    conv_z5v = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(bn_y5v)
    conc5zv = Concatenate(axis=4)([out_c4v, conv_z5v])
    bn_z5v = BatchNormalization()(conc5zv)
    mp_z5v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_z5v)
    
    st_y5v = Conv3D(filters=128, kernel_size=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4v)
    conc5stv = Concatenate(axis=4)([out_c4v, st_y5v])
    bn_st5v = BatchNormalization()(conc5stv)
    mp_st5v = MaxPooling3D(pool_size=(2,2,2), padding='same')(bn_st5v)
    
    out_5v = add([mp_st5v, mp_z5v])
    out_c5v = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_5v)
    out_c5v = BatchNormalization()(out_c5v)
    
    out_c5v = channel_spatial_squeeze_excite(out_c5v, 16)
    
    gap_3D = GlobalAveragePooling3D()(out_c5v)
    final_out = Dense(6, activation='softmax', name='AV_OUT')(gap_3D)
    
    model = Model(inputs=[video_input], outputs=[final_out])
    model.summary()
    
    model.load_weights('face_net_wt_best.h5', by_name = True)
    print("The pre-trained weight loaded.")
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=1e-4),metrics=["accuracy"]) 
    
    return model
