import numpy as np
import cv2
IMAGE_NET_MEANS = [0.485, 0.456, 0.406]
IMAGE_NET_SDS = [0.229, 0.224, 0.225]

# NOTE: it's not unlikely that these functions will result in too much memory consumption
# in that case, try doing everything by modifying the argument, rather than returning a new variable

# make sure that only rgb images are passed
# no checks are conducted to improve performance
def standardise_image(rgb_image: np.ndarray):
    float_image = rgb_image.astype(np.float32)
    # scaling the data into (0, 1)
    float_image /= 255
    # multiplying the denominator by 3, to squeeze the data into, more or less, (-1, 1) range
    float_image[:, :, 0] = (float_image[:, :, 0] - IMAGE_NET_MEANS[0]) / (3*IMAGE_NET_SDS[0])
    float_image[:, :, 1] = (float_image[:, :, 1] - IMAGE_NET_MEANS[1]) / (3*IMAGE_NET_SDS[1])
    float_image[:, :, 2] = (float_image[:, :, 2] - IMAGE_NET_MEANS[2]) / (3*IMAGE_NET_SDS[2])
    return float_image


def restore_standardised_image(standardised_image: np.ndarray):
    manipultaed_image = standardised_image.copy()
    manipultaed_image[:, :, 0] *= 3*IMAGE_NET_SDS[0]
    manipultaed_image[:, :, 1] *= 3*IMAGE_NET_SDS[1]
    manipultaed_image[:, :, 2] *= 3*IMAGE_NET_SDS[2]

    manipultaed_image[:, :, 0] += IMAGE_NET_MEANS[0]
    manipultaed_image[:, :, 1] += IMAGE_NET_MEANS[1]
    manipultaed_image[:, :, 2] += IMAGE_NET_MEANS[2]
    manipultaed_image *= 255

    return manipultaed_image.astype(np.uint8)
