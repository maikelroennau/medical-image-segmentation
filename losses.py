import tensorflow as tf
from tensorflow.keras import backend


def dice_coef(y_true, y_pred, smooth=1e-7):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)
    return tf.keras.backend.mean((2. * intersection + smooth) / (union + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def jaccard_index(y_true, y_pred, smooth=1e-7):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)
    return (intersection + smooth) / ((union + smooth) - (intersection + smooth))


def jaccard_index_loss(y_true, y_pred):
    return 1. - jaccard_index(y_true, y_pred)


def weighted_categorical_crossentropy(y_true, y_pred):
    weights = [0.1, 1., 2.]
    y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    loss = y_true * tf.keras.backend.log(y_pred) * weights
    loss = -tf.keras.backend.sum(loss, -1)
    return loss
