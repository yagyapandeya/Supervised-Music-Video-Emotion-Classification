
from keras import optimizers
from keras.layers import Input
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D
from keras.layers import  BatchNormalization, ReLU, Concatenate, Dense
from keras.layers.merge import concatenate, add
from keras.models import Model

from SE_2DBlock import channel_spatial_squeeze_excite2D

def audio_net(audio_shape_fast):
   
    audio_input_slow = Input(shape=audio_shape_fast)
    
    #******************************************** START SLOW BRANCH ********************************************************#   
    # BLOCK 1: AUDIO Slow
    conv_x1a = Conv2D(filters=16, kernel_size=(1, 5), padding='same', activation='relu', kernel_initializer='he_normal')(audio_input_slow)
    conc1xa = Concatenate(axis=3)([audio_input_slow, conv_x1a])
    
    conv_y1a = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc1xa)
    conc1ya = Concatenate(axis=3)([audio_input_slow, conv_y1a])
    
    conv_z1a = Conv2D(filters=16, kernel_size=(1, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc1ya)
    conc1za = Concatenate(axis=3)([audio_input_slow, conv_z1a])
    mp_z1a = MaxPooling2D(pool_size=(2,2), padding='same')(conc1za)
    
    st_y1a = Conv2D(filters=16, kernel_size=(3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(audio_input_slow)
    conc1sta = Concatenate(axis=3)([audio_input_slow, st_y1a])
    mp_st1a = MaxPooling2D(pool_size=(2,2), padding='same')(conc1sta)
    
    out_1a = add([mp_st1a, mp_z1a])
    out_c1a = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_1a)
    out_c1ax = BatchNormalization()(out_c1a)
    
    out_c1ax = channel_spatial_squeeze_excite2D(out_c1ax, 4)
    
    # BLOCK 2: AUDIO
    conv_x2a = Conv2D(filters=32, kernel_size=(1, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1ax)
    conc2xa = Concatenate(axis=3)([out_c1ax, conv_x2a])
    
    conv_y2a = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc2xa)
    conc2ya = Concatenate(axis=3)([out_c1ax, conv_y2a])
    
    conv_z2a = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc2ya)
    conc2za = Concatenate(axis=3)([out_c1ax, conv_z2a])
    mp_z2a = MaxPooling2D(pool_size=(2,2), padding='same')(conc2za)
    
    st_y2a = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu',kernel_initializer='he_normal')(out_c1ax)
    conc2sta = Concatenate(axis=3)([out_c1ax, st_y2a])
    mp_st2a = MaxPooling2D(pool_size=(2,2), padding='same')(conc2sta)
    
    out_2a = add([mp_st2a, mp_z2a])
    out_c2a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_2a)
    out_c2ax = BatchNormalization()(out_c2a)
    
    out_c2ax = channel_spatial_squeeze_excite2D(out_c2ax, 8)
    
    # BLOCK 3: AUDIO
    conv_x3a = Conv2D(filters=64, kernel_size=(1, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2ax)
    conc3xa = Concatenate(axis=3)([out_c2ax, conv_x3a])
    
    conv_y3a = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc3xa)
    conc3ya = Concatenate(axis=3)([out_c2ax, conv_y3a])
    
    conv_z3a = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc3ya)
    conc3za = Concatenate(axis=3)([out_c2ax, conv_z3a])
    mp_z3a = MaxPooling2D(pool_size=(2,2), padding='same')(conc3za)
    
    st_y3a = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2ax)
    conc3sta = Concatenate(axis=3)([out_c2ax, st_y3a])
    mp_st3a = MaxPooling2D(pool_size=(2,2), padding='same')(conc3sta)
    
    out_3a = add([mp_st3a, mp_z3a])
    out_c3a = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_3a)
    out_c3ax = BatchNormalization()(out_c3a)
    
    out_c3ax = channel_spatial_squeeze_excite2D(out_c3ax, 8)
    
    # BLOCK 4: AUDIO
    conv_x4a = Conv2D(filters=128, kernel_size=(1, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3ax)
    conc4xa = Concatenate(axis=3)([out_c3ax, conv_x4a])
    
    conv_y4a = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc4xa)
    conc4ya = Concatenate(axis=3)([out_c3ax, conv_y4a])
    
    conv_z4a = Conv2D(filters=128, kernel_size=(1, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc4ya)
    conc4za = Concatenate(axis=3)([out_c3ax, conv_z4a])
    mp_z4a = MaxPooling2D(pool_size=(2,2), padding='same')(conc4za)
    
    st_y4a = Conv2D(filters=128, kernel_size=(3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3ax)
    conc4sta = Concatenate(axis=3)([out_c3ax, st_y4a])
    mp_st4a = MaxPooling2D(pool_size=(2,2), padding='same')(conc4sta)
    
    out_4a = add([mp_st4a, mp_z4a])
    out_c4a = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_4a)
    out_c4ax = BatchNormalization()(out_c4a)
    
    out_c4ax = channel_spatial_squeeze_excite2D(out_c4ax, 16)
    
    # BLOCK 5: AUDIO
    conv_x5a = Conv2D(filters=256, kernel_size=(1, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4ax)
    conc5xa = Concatenate(axis=3)([out_c4ax, conv_x5a])
    
    conv_y5a = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc5xa)
    conc5ya = Concatenate(axis=3)([out_c4ax, conv_y5a])
    
    conv_z5a = Conv2D(filters=256, kernel_size=(1, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc5ya)
    conc5za = Concatenate(axis=3)([out_c4ax, conv_z5a])
    mp_z5a = MaxPooling2D(pool_size=(2,2), padding='same')(conc5za)
    
    st_y5a = Conv2D(filters=256, kernel_size=(3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4ax)
    conc5sta = Concatenate(axis=3)([out_c4ax, st_y5a])
    #bn_st5a = BatchNormalization()(conc5sta)
    mp_st5a = MaxPooling2D(pool_size=(2,2), padding='same')(conc5sta)
    
    out_5a = add([mp_st5a, mp_z5a])
    out_c5a = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_5a)
    out_c5ax = BatchNormalization()(out_c5a)
    
    out_c5ax = channel_spatial_squeeze_excite2D(out_c5ax, 16)
    
    #*************************************************** END FAST BRABCH ***********************************************************#
    gap_2D_fast = GlobalAveragePooling2D()(out_c5ax)
 
    final_out = Dense(6, activation='softmax', name='AV_OUT')(gap_2D_fast)
    
    model = Model(inputs=[audio_input_slow], outputs=[final_out])
    model.summary()
    
    model.load_weights('music_net_wt_best.h5', by_name = True)
    print("The pre-trained weight loaded.")
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=1e-4),metrics=["accuracy"])  

    
    return model