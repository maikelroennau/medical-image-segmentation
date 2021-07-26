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


def weighted_categorical_crossentropy(y_true, y_pred):
    weights = [.1, 1., 1.]
    y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    loss = y_true * tf.keras.backend.log(y_pred) * weights
    loss = -tf.keras.backend.sum(loss, -1)
    return loss


def categorical_focal_loss(y_true, y_pred, alpha=[.1, 1., 1.], gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)

    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """
    alpha = tf.constant(alpha, dtype=tf.float32)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.keras.backend.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * tf.keras.backend.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
