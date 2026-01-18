"""
TensorFlow/Keras compatibility module for TensorFlow 2.18+ with segmentation-models.

This module patches compatibility issues between older libraries (efficientnet, 
image-classifiers) and newer versions of TensorFlow/Keras.

Import this module FIRST before any other TensorFlow or segmentation-models imports.

Usage:
    import utils.tf_compat  # This patches the compatibility issues
    import segmentation_models as sm
    import tensorflow as tf
"""

import os
import sys

# Set environment variables before any TensorFlow imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

# Import tf_keras FIRST - this is the Keras 2 API compatibility layer for TF 2.16+
import tf_keras


class GenericUtilsCompat:
    """Compatibility shim for keras.utils.generic_utils"""
    
    @staticmethod
    def get_custom_objects():
        """Return custom objects dictionary from tf_keras"""
        return tf_keras.utils.get_custom_objects()


def patch_keras_for_efficientnet():
    """
    Patch keras module to add missing generic_utils for efficientnet compatibility.
    
    The efficientnet package calls keras.utils.generic_utils.get_custom_objects()
    which was removed in Keras 3. This function adds the missing attribute.
    """
    try:
        import keras
        import keras.utils
        
        # Add generic_utils attribute to keras.utils if it doesn't exist
        if not hasattr(keras.utils, 'generic_utils'):
            keras.utils.generic_utils = GenericUtilsCompat()
            
        # Also ensure get_custom_objects is available directly on keras.utils
        if not hasattr(keras.utils, 'get_custom_objects'):
            keras.utils.get_custom_objects = tf_keras.utils.get_custom_objects
            
    except ImportError as e:
        print(f"Warning: Could not patch keras: {e}")
    except Exception as e:
        print(f"Warning: Error patching keras: {e}")


def patch_keras_backend():
    """Patch keras.backend for image_classifiers compatibility."""
    try:
        import keras.backend as K
        
        # Add image_data_format if missing
        if not hasattr(K, 'image_data_format'):
            K.image_data_format = lambda: tf_keras.backend.image_data_format()
            
    except (ImportError, AttributeError):
        pass


# Apply patches before anything else imports keras
patch_keras_for_efficientnet()
patch_keras_backend()

# Now import tensorflow
import tensorflow as tf

# Suppress TensorFlow info messages
tf.get_logger().setLevel('WARNING')
