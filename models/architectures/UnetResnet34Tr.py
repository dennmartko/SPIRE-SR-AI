import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Flatten, Conv2DTranspose


def UnetResnet34Tr(shape, data_format, C1=64, multipliers=(1, 2, 4, 8)):
    axis = 1 if shape[0] != shape[1] else 3
    # Block parameters
    shape = shape
    conv_params = lambda n, regularize_bool: {'filters':n*C1, 'kernel_initializer':tf.keras.initializers.HeUniform, 'bias_initializer':'zeros', 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_regularizer':'l1_l2' if regularize_bool else None}

    deconv_params = lambda n, pad, regularize_bool: {'filters':n*C1, 'kernel_initializer':tf.keras.initializers.HeUniform, 'bias_initializer':'zeros', 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_regularizer':'l1_l2' if regularize_bool else None}
    
    bn_params = {'momentum':0.9, 'epsilon':1e-5, 'axis':axis}
    drop_params = lambda d: {'rate':d}

    # Model Blocks
    inp = Input(shape=shape)

    x = StemBlock(conv_params(multipliers[0], False), bn_params, inp) # 256x256
    skip_stem = CAM(num_channels=x.shape[axis], ID=0, axis=axis, data_format=data_format, name=f"CAM_{0}")(x)

    # ENCODER BLOCKS
    x = EncoderBlock(conv_params(multipliers[0], False), bn_params, drop_params(0.), 3, x) # 128x128
    skip1 = CAM(num_channels=x.shape[axis], ID=1, axis=axis, data_format=data_format, name=f"CAM_{1}")(x)
    x = EncoderBlock(conv_params(multipliers[1], False), bn_params, drop_params(0.), 4, x) # 64x64
    skip2 = PCAM(num_channels=x.shape[axis], ID=2, axis=axis, data_format=data_format, name=f"PCAM_{0}")(x)
    x = EncoderBlock(conv_params(multipliers[2], False), bn_params, drop_params(0.), 6, x) # 32x32
    skip3 = PCAM(num_channels=x.shape[axis], ID=3, axis=axis, data_format=data_format, name=f"PCAM_{1}")(x)
    x = EncoderBlock(conv_params(multipliers[3], False), bn_params, drop_params(0.), 3, x) # 16x16
    skip4 = PCAM(num_channels=x.shape[axis], ID=4, axis=axis, data_format=data_format, name=f"PCAM_{2}")(x)
    x = EncoderBlock(conv_params(multipliers[3], False), bn_params, drop_params(0.), 2, x) # 8x8
    x = PCAM(num_channels=x.shape[axis], ID=5, axis=axis, data_format=data_format, name=f"PCAM_{3}")(x)

    # UPSAMPLE
    x = Upsample(deconv_params(multipliers[3], False, False), r=2)(x) # 16x16

    # DECODER BLOCKS
    x = tf.concat([x, skip4], axis=axis) # 16x16
    x = DecoderBlock(conv_params(multipliers[3], False), deconv_params(multipliers[2], False, False), bn_params, x) # 32x32
    x = tf.concat([x, skip3], axis=axis) # 32x32
    x = DecoderBlock(conv_params(multipliers[2], False), deconv_params(multipliers[1], False, False), bn_params, x) # 64x64
    x = tf.concat([x, skip2], axis=axis) # 64x64
    x = DecoderBlock(conv_params(multipliers[1], False), deconv_params(multipliers[0], False, False), bn_params, x) # 128x128
    x = tf.concat([x, skip1], axis=axis) # 128x128
    x = DecoderBlock(conv_params(multipliers[0], False), deconv_params(multipliers[0], False, False), bn_params, x) # 256x256

    # concat stem
    x = tf.concat([x, skip_stem], axis=axis) # 256x256
    
    # Conv - Conv - OUT
    x = Conv2D(**conv_params(multipliers[0], False), kernel_size=(3, 3), strides=(1,1))(x) # 256x256
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    out = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), use_bias=True, padding='same', data_format=data_format)(x)
    out = tf.keras.activations.linear(out)
    return tf.keras.Model(inp, out)


class SpatialSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_channels, name=None):
        super(SpatialSelfAttention, self).__init__(name=name)

        # Define linear transformations for queries, keys, and values
        self.num_channels = num_channels

        self.query_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)
        self.key_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)
        self.value_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)

        # Define scaling factor for dot product
        self.scale_factor = tf.math.sqrt(tf.cast(num_channels, dtype=tf.float32))
        
    def call(self, inputs):
        resh = tf.keras.layers.Reshape((inputs.shape[1] * inputs.shape[2], self.num_channels))
        resh_final = tf.keras.layers.Reshape((inputs.shape[1], inputs.shape[2], self.num_channels))

        q = resh(self.query_convs(inputs))
        k = resh(self.key_convs(inputs))
        v = resh(self.value_convs(inputs))

        # dot-product attention
        attention = tf.matmul(q, k, transpose_b=True) / self.scale_factor
        attention = tf.keras.activations.softmax(attention)

        # Calculate attention output
        output = tf.matmul(attention,v)
        output = resh_final(output)
        return output
    

class ChannelSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_channels, name=None):
        super(ChannelSelfAttention, self).__init__(name=name)

        # Define linear transformations for queries, keys, and values
        self.num_channels = num_channels

        self.query_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)
        self.key_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)
        self.value_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)

        # Define scaling factor for dot product
        self.scale_factor = tf.math.sqrt(tf.cast(num_channels, dtype=tf.float32))

    def call(self, inputs):
        # Reshape for each head
        resh = tf.keras.layers.Reshape((self.num_channels, inputs.shape[1] * inputs.shape[2]))
        # Final reshape layer
        out_resh = tf.keras.layers.Reshape((inputs.shape[1], inputs.shape[2], self.num_channels))

        q = resh(self.query_convs(inputs))
        k = resh(self.key_convs(inputs))
        v = resh(self.value_convs(inputs))

        # dot-product attention
        attention = tf.matmul(q, k, transpose_b=True) / self.scale_factor
        attention = tf.keras.activations.softmax(attention)

        # Calculate attention output
        output = tf.matmul(attention, v)
        output = out_resh(output)
        return output

class PCAM(tf.keras.layers.Layer):
    def __init__(self, num_channels, ID, axis, data_format, name=None):
        super(PCAM, self).__init__(name=name)

        ## Scaling parameters
        self.a = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=True, name=f"a_{ID}")
        self.b = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=True, name=f"b_{ID}")

        # Layer arguments
        self.ID = ID
        self.axis = axis
        self.data_format = data_format
        self.num_channels = num_channels

        # Layers
        self.spat_att = SpatialSelfAttention(num_channels=num_channels, name=f"PSA_{self.ID}")
        self.chan_att = ChannelSelfAttention(num_channels=num_channels, name=f"CSA_{self.ID}")

    def call(self, inp):
        ## Spatial Attention
        x = self.spat_att(inp)

        ## Channel Attention
        y = self.chan_att(inp)  

        ## Fusion
        fusion_chan = tf.keras.layers.Add(name=f"ChanFusion_{self.ID}")([self.a*y, inp])
        fusion_spat = tf.keras.layers.Add(name=f"SpatFusion_{self.ID}")([self.b*x, inp])
        fusion = tf.keras.layers.Add(name=f"Fusion_{self.ID}")([fusion_spat, fusion_chan])
        return fusion
    
class CAM(tf.keras.layers.Layer):
    def __init__(self, num_channels, ID, axis, data_format, name=None):
        super(CAM, self).__init__(name=name)

        ## Scaling parameters
        self.a = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=True, name=f"a_{ID}")

        # Layer arguments
        self.ID = ID
        self.axis = axis
        self.data_format = data_format
        self.num_channels = num_channels

        # Layers
        self.chan_att = ChannelSelfAttention(num_channels=num_channels, name=f"CSA_{self.ID}")

    def call(self, inp):
        ## Channel Attention
        x = self.chan_att(inp)

        ## Fusion
        fusion = tf.keras.layers.Add(name=f"ChanFusion_{self.ID}")([self.a*x, inp])
        
        return fusion
    
class Upsample(tf.keras.layers.Layer):
    def __init__(self, deconv_params, r):
        super(Upsample, self).__init__()

        # Layer arguments
        self.deconv_params = deconv_params.copy()
        self.deconv_params["filters"] = deconv_params["filters"]*r**2 # Increase the number of channels since we need HxWxCr^2 --> HrxWrxC
        self.r = r

        # Layers
        self.conv = Conv2D(**self.deconv_params, kernel_size=(3, 3), strides=(1,1))

    def call(self, inp):
        x = self.conv(inp)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.nn.depth_to_space(x, self.r, data_format="NHWC")
        return x

def StemBlock(conv_params, bn_params, inp):
    x = Conv2D(**conv_params, kernel_size=(7, 7), strides=(1,1))(inp)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

def Res32projBlock(conv_params, bn_params, inp):
    res = inp

    # First the F(x) path
    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(2,2))(inp)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(1,1))(x)
    x = BatchNormalization(**bn_params)(x)

    # Residual path
    res = Conv2D(**conv_params, kernel_size=(1, 1), strides=(2,2))(res)

    x = tf.keras.layers.Add()([res, x])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

def Res32Block(conv_params, bn_params, drop_params, inp):
    # First the F(x) path
    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(1,1))(inp)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Possible dropout
    if drop_params["rate"] > 0.:
        x = tf.keras.layers.Dropout(**drop_params)(x)

    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(1,1))(x)
    x = BatchNormalization(**bn_params)(x)

    x = tf.keras.layers.Add()([inp, x])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

def EncoderBlock(conv_params, bn_params, drop_params, n, inp):
    # First the downsampling
    x = Res32projBlock(conv_params, bn_params, inp)

    # Loop for Res32 blocks
    for i in range(n):
        x = Res32Block(conv_params, bn_params, drop_params, x)

    return x

def DecoderBlock(conv_params, deconv_params, bn_params, inp):
    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(1,1))(inp)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = Upsample(deconv_params, 2)(x) # x2 upsample
    return x