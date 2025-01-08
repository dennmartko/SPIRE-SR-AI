import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Flatten, Conv2DTranspose

def RCAN(shape, data_format, C1=64, multipliers=(1, 2, 4, 8)):
    axis = 1 if shape[0] != shape[1] else 3

    # Input layer
    inp = Input(shape=shape)

    # Downsampling of input images
    filters = 32
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', data_format=data_format)(inp)

    # RIR
    x = ResidualInResidual(filters=filters, num_groups=5, num_rcab_per_group=5)(x)

    # Output block
    out = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', data_format=data_format)(x)
    return tf.keras.Model(inp, out)

class RCAB(tf.keras.layers.Layer):
    def __init__(self, filters, reduction=8):
        super(RCAB, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(filters // reduction)
        self.fc2 = tf.keras.layers.Dense(filters, activation='sigmoid')
        self.multiply = tf.keras.layers.Multiply()
        self.leakyrelu = tf.keras.layers.LeakyReLU()
        self.BN = tf.keras.layers.BatchNormalization(epsilon=1e-6)


    def call(self, inputs):
        # Residual connection
        res = inputs

        # Convolution layers
        x = self.conv1(inputs)
        x = self.leakyrelu(x)
        x = self.BN(x)
        x = self.conv2(x)

        # Channel attention
        w = self.global_pool(x)
        w = self.fc1(w)
        w = self.leakyrelu(w)
        w = self.fc2(w)
        w = tf.expand_dims(w, axis=1)
        w = tf.expand_dims(w, axis=1)
        x = self.multiply([x, w])

        # Add the residual connection
        x = x + res
        return x


class ResidualGroup(tf.keras.layers.Layer):
    def __init__(self, filters, num_rcab, reduction=8):
        super(ResidualGroup, self).__init__()
        self.rcabs = [RCAB(filters, reduction) for _ in range(num_rcab)]
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
        #self.leakyrelu = tf.keras.layers.LeakyReLU()

    def call(self, inputs):
        x = inputs
        # Pass through all RCAB blocks
        for rcab in self.rcabs:
            x = rcab(x)
        
        # Residual connection from input to the output after RCAB blocks
        x = self.conv(x)
        x = x + inputs
        return x
    
class ResidualInResidual(tf.keras.layers.Layer):
    def __init__(self, filters, num_groups, num_rcab_per_group, reduction=8):
        super(ResidualInResidual, self).__init__()
        # Create a list of Residual Groups
        self.groups = [ResidualGroup(filters, num_rcab_per_group, reduction) for _ in range(num_groups)]
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')
        #self.leakyrelu = tf.keras.layers.LeakyReLU()

    def call(self, inputs):
        x = inputs
        # Pass through all Residual Groups
        for group in self.groups:
            x = group(x)
        
        # Final residual connection from input to the output after all groups
        x = self.conv(x)
        x = x + inputs
        return x

class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters, scale_factor):
        super(OutputBlock, self).__init__()
        self.scale_factor = scale_factor
        
        # Upsampling layers
        self.upsample_layer = tf.keras.layers.Conv2D(filters * scale_factor**2, kernel_size=3, padding='same')
        
        # Final convolution layer
        self.conv_final = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')  # Assuming RGB output

    def call(self, inputs):
        x = inputs
        # Apply upsampling
        x = self.upsample_layer(x)

        x = tf.nn.depth_to_space(x, block_size=2, data_format="NHWC")
        # Final 3x3 convolution
        x = self.conv_final(x)
        return x
