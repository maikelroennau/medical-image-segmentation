from tensorflow import keras


def dice_coef(y_true, y_pred, smooth=1.):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = keras.backend.sum(y_true, axis=[1, 2, 3]) + keras.backend.sum(y_pred, axis=[1, 2, 3])
    return keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)