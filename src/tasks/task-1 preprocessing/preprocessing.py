import os
from typing import Optional
import numpy as np
import cv2

def read_images(
    path,
    extensions: Optional[set[str]] = ('jpg', 'png', 'jpeg') )\
        -> list[np.ndarray]:
    """
    Reads all images from a directory.

    :param path: Path to the directory.
    :param extensions: Set of allowed extensions.
    :return: List of images.
    """
    #Find the Images with the Extenstion in the Tupple

    images_array = [cv2.imread(path +"//"+ file_)
        for file_ in os.listdir(path)
        if file_.endswith(tuple(extensions))]

    return images_array

def rescale_images(
    images: list[np.ndarray],
    new_shape: tuple[int, int])\
        -> list[np.ndarray]:
    """
    Rescales images to a new shape.

    :param images: List of images.
    :param new_shape: New shape.
    """

    return [cv2.resize(im, new_shape) for im in images]

def _standardize_image(
    image: np.ndarray,
    ) -> np.ndarray:
    """
    Standardizes an image.
    center the image, and normalize the pixel values to be between 0 and 1.

    :param image: Image to standardize.
    :return: Standardized image.
    """
    mean = np.mean(image[:])
    std = np.std(image[:])

    return (image - mean) / std

def standardize_images(
    images: list[np.ndarray],
    ) -> list[np.ndarray]:
    """
    Standardizes images.
    
    :param images: List of images.
    :return: List of standardized images.
    """
    return list(map(_standardize_image, images))

def _normalize_image(
    image: np.ndarray,
    ) -> np.ndarray:
    """
    Normalizes an image.

    :param image: Image.
    :return: Normalized image.
    """
    return image / 255.0

def normalize_images(
    images: list[np.ndarray],
    ) -> list[np.ndarray]:
    """
    Normalizes images.

    :param images: List of images.
    :return: List of normalized images.
    """
    return list(map(_normalize_image, images))

def _clip_image(
    image: np.ndarray,
    ) -> np.ndarray:
    """
    Clips an image.

    :param image: Image.
    :return: Clipped image.
    """
    return np.clip(image, 0.0, 1.0)

def clip_images(
    images: list[np.ndarray],
    ) -> list[np.ndarray]:
    """
    Clips images.

    :param images: List of images.
    :return: List of clipped images.
    """
    return list(map(_clip_image, images))

def convert_images_to_gray(
    images: list[np.ndarray],
    ) -> list[np.ndarray]:
    """
    Converts images to grayscale.

    :param images: List of images.
    :return: List of grayscale images.
    """
    return list(map(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), images))

# preprocess pipeline
def preprocess_images(
    images: list[np.ndarray],
    rescaled_shape: [tuple[int, int]],
    ) -> list[np.ndarray]:
    """
    Preprocesses images.

    :param images: List of images.
    :return: List of preprocessed images.
    """
    # 1 - convert to grayscale
    images = convert_images_to_gray(images)
    # 2 - rescale
    images = rescale_images(images, rescaled_shape)
    # 3 - normalize
    images = normalize_images(images)
    # 4 - standardize
    images = standardize_images(images)
    # 5 - clip
    images = clip_images(images)
    # return list(map(clip_image, standardize_images(normalize_images(images))))
    return images

