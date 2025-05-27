import tensorflow as tf
from tensorflow.keras import layers, Model

# XLA-optimized U-Net with ResNet-34 backbone and Parallel-CAM modules
# All layers defined once in __init__, static loops unrolled, channel-last format

class SpatialSelfAttention(layers.Layer):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.scale = tf.math.sqrt(tf.cast(channels, tf.float32))
        self.q_conv = layers.Conv2D(channels, 1)
        self.k_conv = layers.Conv2D(channels, 1)
        self.v_conv = layers.Conv2D(channels, 1)
        self.reshape_to_seq = layers.Reshape((-1, channels))
        self.reshape_to_map = None

    def build(self, input_shape):
        self.reshape_to_map = layers.Reshape((input_shape[1], input_shape[2], input_shape[3]))
        super().build(input_shape)

    def call(self, x):
        q = self.reshape_to_seq(self.q_conv(x))  # [B, H*W, C]
        k = self.reshape_to_seq(self.k_conv(x))
        v = self.reshape_to_seq(self.v_conv(x))
        attn = tf.matmul(q, k, transpose_b=True) / self.scale
        attn = tf.nn.softmax(attn)
        out = tf.matmul(attn, v)
        return self.reshape_to_map(out)
    
class ChannelSelfAttention(layers.Layer):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.scale = tf.math.sqrt(tf.cast(channels, tf.float32))
        self.q_conv = layers.Conv2D(channels, 1)
        self.k_conv = layers.Conv2D(channels, 1)
        self.v_conv = layers.Conv2D(channels, 1)
        self.reshape_to_seq = layers.Reshape((channels, -1))
        self.reshape_to_map = None

    def build(self, input_shape):
        self.reshape_to_map = layers.Reshape((input_shape[1], input_shape[2], input_shape[3]))
        super().build(input_shape)

    def call(self, x):
        q = self.reshape_to_seq(self.q_conv(x))  # [B, C, H*W]
        k = self.reshape_to_seq(self.k_conv(x))
        v = self.reshape_to_seq(self.v_conv(x))
        attn = tf.matmul(q, k, transpose_b=True) / self.scale
        attn = tf.nn.softmax(attn)
        out = tf.matmul(attn, v)
        return self.reshape_to_map(out)


class PCAM(layers.Layer):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.alpha = self.add_weight('alpha', shape=(), initializer='zeros')
        self.beta = self.add_weight('beta', shape=(), initializer='zeros')
        self.spatial = SpatialSelfAttention(channels)
        self.channel = ChannelSelfAttention(channels)

    def call(self, x):
        sa = self.spatial(x)
        ca = self.channel(x)
        out = x + self.beta * sa + self.alpha * ca
        return out

class CAM(layers.Layer):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.alpha = self.add_weight('alpha', shape=(), initializer='zeros')
        self.channel = ChannelSelfAttention(channels)

    def call(self, x):
        ca = self.channel(x)
        return x + self.alpha * ca

class ResBlock(layers.Layer):
    def __init__(self, channels, dropout_rate=0.0, name=None):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(channels, 3, padding='same', kernel_initializer='he_uniform', use_bias=True)
        self.bn1 = layers.BatchNormalization(epsilon=1e-6, momentum=0.9)
        self.act1 = layers.LeakyReLU(0.2)
        self.drop = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.conv2 = layers.Conv2D(channels, 3, padding='same', kernel_initializer='he_uniform', use_bias=True)
        self.bn2 = layers.BatchNormalization(epsilon=1e-6, momentum=0.9)
        self.act2 = layers.LeakyReLU(0.2)

    def call(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        if self.drop: y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return self.act2(x + y)

class ResProjBlock(layers.Layer):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(channels, 3, strides=2, padding='same', kernel_initializer='he_uniform', use_bias=True)
        self.bn1 = layers.BatchNormalization(epsilon=1e-6, momentum=0.9)
        self.act1 = layers.LeakyReLU(0.2)
        self.conv2 = layers.Conv2D(channels, 3, padding='same', kernel_initializer='he_uniform')
        self.bn2 = layers.BatchNormalization(epsilon=1e-6, momentum=0.9)
        self.proj = layers.Conv2D(channels, 1, strides=2, padding='same', kernel_initializer='he_uniform', use_bias=True)
        self.act2 = layers.LeakyReLU(0.2)

    def call(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        r = self.proj(x)
        return self.act2(r + y)

class Upsample(layers.Layer):
    def __init__(self, channels, scale=2, name=None):
        super().__init__(name=name)
        self.conv = layers.Conv2D(channels * (scale**2), 3, padding='same', kernel_initializer='he_uniform', use_bias=True)
        self.act = layers.LeakyReLU(0.2)
        self.scale = scale

    def call(self, x):
        x = self.act(self.conv(x))
        return tf.nn.depth_to_space(x, self.scale)

class UnetResnet34TrNew(Model):
    def __init__(self, input_shape, data_format, C1=64, multipliers=(1,2,4,8)):
        super().__init__()
        c1 = C1
        self.stem = tf.keras.Sequential([
            layers.Conv2D(c1, 7, padding='same', kernel_initializer='he_uniform', use_bias=True),
            layers.BatchNormalization(epsilon=1e-6, momentum=0.9),
            layers.LeakyReLU(0.2)
        ])

        # Encoder
        self.enc1 = self._make_encoder_stage(c1 * multipliers[0], 3)
        self.enc2 = self._make_encoder_stage(c1 * multipliers[1], 4)
        self.enc3 = self._make_encoder_stage(c1 * multipliers[2], 6)
        self.enc4 = self._make_encoder_stage(c1 * multipliers[3], 3)
        self.bottleneck = self._make_encoder_stage(c1 * multipliers[3], 2)
        # # Attention
        self.attn_stem = CAM(c1)
        self.attn1 = CAM(c1 * multipliers[0])
        self.attn2 = PCAM(c1 * multipliers[1])
        self.attn3 = PCAM(c1 * multipliers[2])
        self.attn4 = PCAM(c1 * multipliers[3])
        self.attn_bottleneck = PCAM(c1 * multipliers[3])
        # # Decoder
        self.up4 = Upsample(c1 * multipliers[3])
        self.dec4 = self._make_decoder_stage(c1 * multipliers[3], c1 * multipliers[2])
        self.dec3 = self._make_decoder_stage(c1 * multipliers[2], c1 * multipliers[1])
        self.dec2 = self._make_decoder_stage(c1 * multipliers[1], c1 * multipliers[0])
        self.dec1 = self._make_decoder_stage(c1 * multipliers[0], c1 * multipliers[0])
        # # Final conv
        self.final_conv = tf.keras.Sequential([
            layers.Conv2D(c1, 3, padding='same', kernel_initializer='he_uniform', use_bias=True),
            layers.BatchNormalization(epsilon=1e-6, momentum=0.9),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, 3, padding='same', kernel_initializer='he_uniform', use_bias=True),
        ])

    def _make_encoder_stage(self, channels, num_blocks, use_PCAM=True):
        layers_list = [ResProjBlock(channels)]
        for _ in range(num_blocks):
            layers_list.append(ResBlock(channels))
        return tf.keras.Sequential(layers_list)

    def _make_decoder_stage(self, in_ch, out_ch):
        return tf.keras.Sequential([
            layers.Conv2D(in_ch, 3, padding='same', kernel_initializer='he_uniform', use_bias=True),
            layers.BatchNormalization(epsilon=1e-6, momentum=0.9),
            layers.LeakyReLU(0.2),
            Upsample(out_ch)
        ])

    def call(self, x):
        # Encoder
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        b  = self.bottleneck(x4)
        # Decoder with skip connections
        up = self.up4(self.attn_bottleneck(b))
        d4 = tf.concat([up, self.attn4(x4)], axis=-1)
        d4 = self.dec4(d4)
        d3 = tf.concat([d4, self.attn3(x3)], axis=-1)
        d3 = self.dec3(d3)
        d2 = tf.concat([d3, self.attn2(x2)], axis=-1)
        d2 = self.dec2(d2)
        d1 = tf.concat([d2, self.attn1(x1)], axis=-1)
        d1 = self.dec1(d1)

        # #Final conv
        out = tf.concat([d1, self.attn_stem(x0)], axis=-1)
        out = self.final_conv(out)
        return out