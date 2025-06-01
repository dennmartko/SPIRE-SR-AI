import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Activation

def conv_block(input_tensor, filters, kernel_size, strides, use_bias, data_format, momentum, alpha, dropout_rate=None):
    x = Conv2D(filters, (kernel_size, kernel_size), strides=strides, use_bias=use_bias, 
               padding='same', data_format=data_format)(input_tensor)
    x = BatchNormalization(momentum=momentum, epsilon=1e-4)(x)
    x = LeakyReLU(alpha=alpha)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def conv_transpose_block(input_tensor, filters, kernel_size, strides, use_bias, data_format, 
                         momentum, alpha, dropout_rate=None, output_padding=None):
    x = Conv2DTranspose(filters, (kernel_size, kernel_size), strides=strides, use_bias=use_bias, 
                        padding='same', output_padding=output_padding, data_format=data_format)(input_tensor)
    x = BatchNormalization(momentum=momentum, epsilon=1e-4)(x)
    x = LeakyReLU(alpha=alpha)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def build_unet(shape, data_format):
    axis = 1 if shape[0] != shape[1] else 3
    C1, K1, d, B1, B2, LR, momentum = 64, 4, 0.5, True, True, 0.2, 0.9

    inp = Input(shape=shape)

    # Encoder (Downsampling to 8x8 spatial resolution)
    act1 = conv_block(inp, C1, K1, 2, B1, data_format, momentum, LR)  # Down to 128x128
    act2 = conv_block(act1, 2 * C1, K1, 2, B1, data_format, momentum, LR)  # Down to 64x64
    act3 = conv_block(act2, 4 * C1, K1, 2, B1, data_format, momentum, LR)  # Down to 32x32
    act4 = conv_block(act3, 8 * C1, K1, 2, B1, data_format, momentum, LR)  # Down to 16x16
    act5 = conv_block(act4, 16 * C1, K1, 2, B1, data_format, momentum, LR, d)  # Down to 8x8

    # Decoder (Upsampling back to 256x256)
    drop6 = conv_transpose_block(act5, 16 * C1, K1, 2, B1, data_format, momentum, LR, d)  # Up to 16x16
    concat4 = tf.concat([drop6, act4], axis=axis)

    drop7 = conv_transpose_block(concat4, 8 * C1, K1, 2, B1, data_format, momentum, LR, d)  # Up to 32x32
    concat3 = tf.concat([drop7, act3], axis=axis)

    drop8 = conv_transpose_block(concat3, 4 * C1, K1, 2, B1, data_format, momentum, LR, None)  # Up to 64x64
    concat2 = tf.concat([drop8, act2], axis=axis)

    drop9 = conv_transpose_block(concat2, 2 * C1, K1, 2, B1, data_format, momentum, LR)  # Up to 128x128
    concat1 = tf.concat([drop9, act1], axis=axis)

    act10 = conv_transpose_block(concat1, C1, K1, 2, B1, data_format, momentum, LR)  # Up to 256x256

    # Final Layer
    lay11_1 = Conv2DTranspose(1, (K1, K1), strides=(1, 1), use_bias=B2, padding='same', data_format=data_format)(act10)
    act11 = Activation('sigmoid')(lay11_1) # must be removed for better performance

    # Model
    model = tf.keras.Model(inp, act11)
    return model
