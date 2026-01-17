import tensorflow as tf
import numpy as np

def l2_normalize_quat(q):
    return tf.math.l2_normalize(q, axis=-1)

def quat_conjugate(q):
    w, x, y, z = tf.unstack(q, axis=-1)
    return tf.stack([w, -x, -y, -z], axis=-1)

def quat_hamilton_product(q1, q2):
    w1, x1, y1, z1 = tf.unstack(q1, axis=-1)
    w2, x2, y2, z2 = tf.unstack(q2, axis=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return tf.stack([w, x, y, z], axis=-1)

def quaternion_loss_QME(y_true, y_pred):
    y_pred = l2_normalize_quat(y_pred)
    q_err  = quat_hamilton_product(y_true, quat_conjugate(y_pred))
    vec = q_err[..., 1:4]
    return tf.reduce_sum(tf.abs(tf.clip_by_value(vec, -1.0, 1.0)), axis=-1)

def angular_error_deg(y_true, y_pred):
    y_pred = l2_normalize_quat(y_pred)
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    dot = tf.clip_by_value(tf.abs(dot), 0.0, 1.0)
    ang = 2.0 * tf.acos(dot)
    return ang * (180.0 / np.pi)
