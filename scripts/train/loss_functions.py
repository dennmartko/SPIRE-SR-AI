import tensorflow as tf

@tf.function(jit_compile=True)
def huber_loss(predictions, y):
    x = y - predictions
    return tf.reduce_sum(x + tf.math.softplus(-2.0 * x) - tf.cast(tf.math.log(2.0), x.dtype))/tf.cast(tf.shape(y)[0], tf.float32)

@tf.function(jit_compile=True)
def aper_loss(predictions, y, aper_mask):
    return tf.reduce_sum(aper_mask*tf.abs(predictions - y))/tf.cast(tf.shape(y)[0], tf.float32)

# Function used during weight training
@tf.function(jit_compile=True)
def old_non_adversarial_loss(predictions, y, aper_mask, alpha):
    Lh = huber_loss(predictions, y)
    Laper = aper_loss(predictions, y, aper_mask)
    return Lh + 5e-3*Laper

# Function used during weight training
@tf.function(jit_compile=True)
def non_adversarial_loss(predictions, y, aper_mask, alpha):
    Lh = huber_loss(predictions, y)
    Laper = aper_loss(predictions, y, aper_mask)
    return alpha*Lh + (1-alpha)*Laper