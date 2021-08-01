import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1):
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
    weights = [0.5, 1.8, 2.]
    y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    loss = y_true * tf.keras.backend.log(y_pred) * weights
    loss = -tf.keras.backend.sum(loss, -1)
    return loss


def categorical_focal_loss(y_true, y_pred, alpha=[0.0136, 98.7, 99.94], gamma=2.):
    """Source: https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
            m
        FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
            c=1
        where m = number of classes, c = class and o = observation
    Parameters:
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :param alpha: The same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
        categories/labels, the size of the array needs to be consistent with the number of classes.
        :param gamma: Focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
        model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
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


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    """Source: https://github.com/mlyg/unified-focal-loss/blob/main/loss-functions.py
    """
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.25, gamma=2.):
    """Source: https://github.com/mlyg/unified-focal-loss/blob/main/loss-functions.py
    """
    def loss_function(y_true, y_pred):
        """For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.25
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """
        axis = identify_axis(y_true.get_shape())

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)

	    #calculate losses separately for each class, only suppressing background class
        back_ce = tf.keras.backend.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = tf.keras.backend.mean(tf.keras.backend.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """Source: https://github.com/mlyg/unified-focal-loss/blob/main/loss-functions.py
    This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = tf.keras.backend.sum(y_true * y_pred, axis=axis)
        fn = tf.keras.backend.sum(y_true * (1-y_pred), axis=axis)
        fp = tf.keras.backend.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0])
        fore_dice = (1-dice_class[:,1]) * tf.keras.backend.pow(1-dice_class[:,1], -gamma)

        # Sum up classes to one score
        loss = tf.keras.backend.mean(tf.keras.backend.sum(tf.stack([back_dice,fore_dice],axis=-1), axis=-1))

        # adjusts loss to account for number of classes
        num_classes = tf.keras.backend.cast(tf.keras.backend.shape(y_true)[-1],'float32')
        loss = loss / num_classes
        return loss

    return loss_function


################################
#      Unified Focal loss      #
################################
def unified_focal_loss(y_true, y_pred, weight=0.5, delta=0.6, gamma=0.2):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framewortf.keras.backend.
    Parameters
    ----------
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    weight : float, optional
        represents lambda parameter and controls weight given to Asymmetric Focal Tversky loss and Asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.2
    """
    # Obtain Asymmetric Focal Tversky loss
    asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
    # Obtain Asymmetric Focal loss
    asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
    # return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
    if weight is not None:
        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)
    else:
        return asymmetric_ftl + asymmetric_fl
