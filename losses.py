import tensorflow as tf


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
