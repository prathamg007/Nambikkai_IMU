import tensorflow as tf
from tensorflow.keras import layers, Model, Input
try:
    from tensorflow_addons.activations import mish
except Exception:
    mish = tf.nn.mish

def conv_stack(x, filters=128, kernel=11, pool=3, noise_std=0.05, dropout=0.25):
    x = layers.GaussianNoise(noise_std)(x)
    x = layers.Conv1D(filters, kernel, padding="same")(x)
    x = layers.Activation(mish)(x)
    x = layers.MaxPooling1D(pool)(x)
    x = layers.Dropout(dropout)(x)
    return x

def build_modelA(window_size: int) -> Model:
    acc_in = Input(shape=(window_size, 3), name="acc_in")
    gyr_in = Input(shape=(window_size, 3), name="gyr_in")
    fs_in  = Input(shape=(1,), name="fs_in")

    a = conv_stack(acc_in); a = conv_stack(a)
    a = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(a); a = layers.Dropout(0.25)(a)

    g = conv_stack(gyr_in); g = conv_stack(g)
    g = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(g); g = layers.Dropout(0.25)(g)

    x = layers.Concatenate()([a, g])
    x = layers.Dense(512, activation=mish)(x)

    f = layers.Dense(512, activation=mish)(fs_in)
    x = layers.Concatenate()([x, f])
    x = layers.Dense(256, activation=mish)(x)
    out = layers.Dense(4, activation=None)(x)
    out = layers.Lambda(lambda q: tf.math.l2_normalize(q, axis=-1), name="unit_quat")(out)
    return Model(inputs=[gyr_in, acc_in, fs_in], outputs=out, name="ModelA_Quat")
