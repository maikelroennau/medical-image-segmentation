import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    return tf.keras.backend.mean((2 * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jaccard_index(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.abs(tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3]))
    union = tf.keras.backend.abs(tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3]))
    return (intersection + smooth) / ((union + smooth) - (intersection + smooth))


def jaccard_index_loss(y_true, y_pred):
    return 1 - jaccard_index(y_true, y_pred)
