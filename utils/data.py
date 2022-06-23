from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from skimage.io import imread
from tqdm import tqdm


SUPPORTED_IMAGE_TYPES = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def reset_class_values(mask: np.ndarray) -> np.ndarray:
    """Reset the mask values corresponding to classes to start from zero.

    Args:
        mask (np.ndarray): The mask to have its values reset.

    Returns:
        np.ndarray: The reset mask.
    """
    reset_image = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i in range(mask.shape[-1]):
        reset_image[:, :][mask[:, :, i] > 0] = i

    return reset_image


def one_hot_encode(image: Union[np.ndarray, tf.Tensor], classes: int, as_numpy: Optional[bool] = False) -> tf.Tensor:
    """Converts an image to the one-hot-encode format.

    Args:
        image (Union[np.ndarray, tf.Tensor]): The image to be converted.
        classes (int): The number of classes in the image. Each class will be put in a dimension.
        as_numpy: (Optional[bool], optional): Whether to return the listed files as Numpy comparable objects.

    Raises:
        TypeError: If the type of `image` is not `np.ndarray` or `tf.Tensor`.

    Returns:
        tf.Tensor: The one-hot-encoded image.
    """
    if isinstance(image, np.ndarray):
        one_hot_encoded = np.zeros(image.shape[:2] + (classes,), dtype=np.uint8)
        for i, unique_value in enumerate(np.unique(image)):
            one_hot_encoded[:, :, i][image == unique_value] = 1
        return one_hot_encoded
    elif isinstance(image, tf.Tensor):
        image = tf.cast(image, dtype=tf.int32)
        image = tf.one_hot(image, depth=classes, axis=2, dtype=tf.int32)
        image = tf.squeeze(image)
        image = tf.cast(image, dtype=tf.float32)

        if as_numpy:
            image = image.numpy()

        return image
    else:
        raise TypeError(f"Argument `image` should be `np.ndarray` or `tf.Tensor. Given `{type(image)}`.")


def load_image(
    image_path: Union[str, tf.Tensor],
    shape: Tuple[int, int] = None,
    normalize: Optional[bool] = False,
    as_gray: Optional[bool] = False,
    as_numpy: Optional[bool] = False) -> tf.Tensor:
    """Read an image from the storage media.

    Args:
        image_path (Union[str, tf.Tensor]): The path to the image file.
        shape (Tuple[int, int], optional): Shape the read image should have after reading it, in the format (HEIGHT, WIDTH). If `None`, the shape will not be changed. Defaults to None.
        normalize (Optional[bool], optional): Whether or not to put the image values between zero and one ([0,1]). Defaults to False.
        as_gray (Optional[bool], optional): Whether or not to read the image in grayscale. Defaults to False.
        as_numpy: (Optional[bool], optional): Whether to return the listed files as Numpy comparable objects.

    Raises:
        TypeError: If the image type is not supported.
        FileNotFoundError: If `image_path` does not point to a file.
        TypeError: If `image_path` is not a `str` or `tf.Tensor` object.

    Returns:
        tf.Tensor: The read image.
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)

        if image_path.is_file():
            if image_path.suffix not in SUPPORTED_IMAGE_TYPES:
                raise TypeError(f"Image type `{image_path.suffix}` not supported.\
                    The supported types are `{', '.join(SUPPORTED_IMAGE_TYPES)}`.")
            else:
                channels = 1 if as_gray else 3

                if image_path.suffix in [".tif", ".tiff"]:
                    image = imread(str(image_path), as_gray=as_gray)
                    image = tf.convert_to_tensor(image, dtype=tf.float32)
                else:
                    image = tf.io.read_file(str(image_path))
                    image = tf.image.decode_png(image, channels=channels)

                if shape:
                    if shape != image.shape[:2]:
                        image = tf.image.resize(image, shape, method="nearest")

                if normalize:
                    image = tf.cast(image, dtype=tf.float32)
                    image = image / 255.

                if as_numpy:
                    image = image.numpy()

                    if as_gray:
                        image = image[:, :, 0]

                return image
        else:
            raise FileNotFoundError(f"The file `{image_path}` was not found.")
    elif isinstance(image_path, tf.Tensor):
        channels = 1 if as_gray else 3
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=channels)

        if shape:
            if shape != image.shape[:2]:
                image = tf.image.resize(image, shape, method="nearest")

        if normalize:
            image = tf.cast(image, dtype=tf.float32)
            image = image / 255.

        return image
    else:
        raise TypeError("Object `{image_path}` is not supported.")


def list_files(
    files_path: str,
    as_numpy: Optional[bool] = False,
    file_types: Optional[list] = SUPPORTED_IMAGE_TYPES,
    seed: Optional[int] = 1234) -> tf.raw_ops.ShuffleDataset:
    """Lists files under `files_path`.

    Args:
        files_path (str): The path to the directory containing the files to be listed.
        as_numpy: (Optional[bool], optional): Whether to return the listed files as Numpy comparable objects.
        seed (Optional[int], optional): A seed used to shuffle the listed files. Note: If listing images and segmentation masks, the same seed must be used. Defaults to 1234.

    Raises:
        RuntimeError: If no files are found under `files_path`.
        FileNotFoundError: If `files_path` is not a directory.

    Returns:
        tf.raw_ops.ShuffleDataset: The listed files.
    """
    if Path(files_path).is_dir():
        patterns = [str(Path(files_path).joinpath(f"*{image_type}")) for image_type in file_types]
        files_list = tf.data.Dataset.list_files(patterns, shuffle=True, seed=seed)

        if len(files_list) == 0:
            raise RuntimeError(f"No files were found at `{files_path}`.")

        if as_numpy:
            files_list = [file.numpy().decode("utf-8") for file in files_list]
            files_list.sort()

        return files_list
    else:
        raise FileNotFoundError(f"The directory `{files_path}` does not exist.")


def load_dataset_files(
    image_path: Union[str, tf.Tensor],
    mask_path: Union[str, tf.Tensor],
    shape: Tuple[int, int],
    classes: int,
    mask_one_hot_encoded: Optional[bool] = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load an image and a segmentation mask.

    Args:
        image_path (Union[str, tf.string, tf.Tensor]): The path to the image file.
        mask_path (Union[str, tf.string, tf.Tensor]): The path to the mask file.
        shape (Tuple[int, int]): The shape the loaded images and mask should have, in the format (HEIGHT, WIDTH).
        classes (int): The number of classes contained in the segmentation mask. It affects the one-hot-encoding of the mask.
        mask_one_hot_encoded (Optional[bool], optional): Whether or not to one-hot-encode the segmentation mask. Defaults to True.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: The loaded image and mask.
    """
    image = load_image(image_path, shape=shape, normalize=True)
    mask = load_image(mask_path, shape=shape, as_gray=True)

    if mask_one_hot_encoded:
        mask = one_hot_encode(mask, classes=classes)

    return image, mask


def load_dataset(
    dataset_path: str,
    batch_size: Optional[int] = 1,
    shape: Tuple[int, int] = (1920, 2560),
    classes: Optional[int] = 3,
    mask_one_hot_encoded: Optional[bool] = True,
    repeat: Optional[bool] = False,
    shuffle: Optional[bool] = True) -> tf.data.Dataset:
    """Loads a `tf.data.Dataset`.

    Args:
        dataset_path (str): The path to the directory containing a subdirectory name `images` and another `masks`.
        batch_size (Optional[int], optional): The number of elements per batch. Defaults to 1.
        shape (Tuple[int, int], optional): The shape the loaded images should have, in the format `(HEIGHT, WIDTH)`. Defaults to (1920, 2560).
        classes (Optional[int], optional): The number of classes in the masks. Defaults to 3.
        mask_one_hot_encoded (Optional[bool], optional): Converts the masks to one-hot-encoded masks, where one dimension is added per class, and each dimension is a binary mask of that class. Defaults to True.
        repeat (Optional[bool], optional): Whether the dataset should be infinite or not. Defaults to False.
        shuffle (Optional[bool], optional): Whether to shuffle the loaded elements. Defaults to True.

    Raises:
        FileNotFoundError: In case the dataset path does not exist.
        FileNotFoundError: In case the directory `images` under the dataset path does not exist.
        FileNotFoundError: In case the directory `masks` under the dataset path does not exist.

    Returns:
        tf.data.Dataset: The loaded dataset.
    """
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        images_path = dataset_path.joinpath("images")
        masks_path = dataset_path.joinpath("masks")

        if not images_path.is_dir():
            raise FileNotFoundError(f"The directory `{str(images_path)}` does not exist.")
        if not masks_path.is_dir():
            raise FileNotFoundError(f"The directory `{str(masks_path)}` does not exist.")

        images_list = list_files(str(images_path))
        masks_list = list_files(str(masks_path))

        dataset = tf.data.Dataset.zip((images_list, masks_list))

        dataset = dataset.map(
            lambda image_path, mask_path: load_dataset_files(
                image_path=image_path,
                mask_path=mask_path,
                shape=shape,
                classes=classes,
                mask_one_hot_encoded=mask_one_hot_encoded
            )
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * batch_size)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset
    else:
        raise FileNotFoundError(f"The directory `{str(dataset_path)}` does not exist.")


def write_dataset(
    dataset: tf.data.Dataset,
    batches: Optional[int] = None,
    output_path: Optional[str] = "dataset_visualization",
    rgb_masks: Optional[bool] = True,
    mask_intensity: Optional[int] = 127) -> None:
    """Write to disk a given `tf.data.Dataset`.

    Args:
        dataset (tf.data.Dataset): The dataset to be written, in the format `((None, HEIGHT, WIDTH, CHANNELS), (None, HEIGHT, WIDTH, CLASSES))`.
        batches (Optional[int], optional): The number of batches of the dataset to write.. Defaults to None.
        output_path (Optional[str], optional): The path where to save the written items of the dataset.. Defaults to "dataset_visualization".
        rgb_masks (Optional[bool], optional): Whether to write the masks as RGB images.. Defaults to True.
        mask_intensity (Optional[int], optional): The intensity of the masks. Defaults to 127.
    """
    output = Path(output_path)
    images_path = output.joinpath("images")
    masks_path = output.joinpath("masks")

    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)

    if not batches:
        try:
            batches = len(dataset)
        except Exception:
            batches = 1
            print("It was not possible to determine the number of batches in the dataset, writing only one.")
    else:
        batches = max(0, batches)

    for i, batch in tqdm(enumerate(dataset), total=batches):
        for j, (image, mask) in enumerate(zip(batch[0], batch[1])):

            image_name = str(images_path.joinpath(f"batch_{i}_image_{j}.png"))
            mask_name = str(masks_path.joinpath(f"batch_{i}_mask_{j}.png"))

            tf.keras.preprocessing.image.save_img(image_name, image)

            if not rgb_masks:
                mask = tf.image.rgb_to_grayscale(mask)
                for new_intensity, intensity in enumerate(np.unique(mask)):
                    mask = np.where(mask == intensity, new_intensity, mask)

            tf.keras.preprocessing.image.save_img(mask_name, mask * mask_intensity, scale=False)

        if i == batches:
            break
