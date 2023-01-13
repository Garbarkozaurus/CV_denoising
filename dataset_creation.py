import numpy as np
import cv2
from pathlib import Path
from typing import Callable
import os

def image_slices(image: np.ndarray, slice_height: int = 256, slice_width: int = 256):
    image_height, image_width, image_channels = image.shape
    assert slice_width <= image_width
    assert slice_height <= image_height
    assert image_channels == 3
    vertical_fits = image_height//slice_height
    horizontal_fits = image_width//slice_width
    top_left_x, top_left_y = 0, 0
    slice_list = []
    for i in range(vertical_fits):
        for j in range(horizontal_fits):
            slice_list.append(image[top_left_y:top_left_y+slice_height, top_left_x:top_left_x+slice_width, :])
            top_left_x += slice_width
        top_left_x = 0
        top_left_y += slice_height
    return slice_list

def image_with_gaussian_noise(image: np.ndarray, mean: int = 0, std: float = 0.05):
    row,col,ch = image.shape
    image_as_int = image.astype(np.int32)
    gauss = np.random.normal(mean,std,(row,col,ch))
    gauss = (255*gauss.reshape(row,col,ch)).astype(np.int32)

    # add noise, but make sure that the values remain in the valid range for np.uint8
    # first, take element-wise maximum between noisy image, and array filled with 0
    # then, take the element-wise minimum of the result and an array filled with 255
    # that way, no overflow can occur after converting the array back to np.uint8
    noisy_image = np.maximum(np.zeros((row, col, ch), dtype=np.int32), image_as_int+gauss)
    noisy_image = np.minimum(np.full((row,col,ch), 255,dtype=np.int32), noisy_image)

    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


# work in progress
def new_sliced_and_add_noisy_directory(source_directory, base_destination_directory: str, slice_height: int = 256, slice_width: int = 256, noise_function: Callable = image_with_gaussian_noise, noisy_image_prefix: str = "GAUSS_"):
    # for all ground-truth images in that directory
    # (int the original SIDD this should always be just a single image)
    for file in source_directory.glob("**/GT_*.PNG"):
        # get relative path from script location to that file
        file_path = str(file)
        # extract the name of the lowest level directory
        directory_name = file_path.split("/")[-2]
        # extract the name of the image
        file_name = file_path.split("/")[-1]
        # split the image name into its "base name" and extension
        # in the "base name", take only the part after "GT_", so that it can be reused
        base_file_name = file_name.split(".")[0].split("GT_")[1]
        extension_file_name = "."+file_name.split(".")[1]

        # load the ground-truth image
        img = cv2.imread(file_path)
        # cut it up into slices of given dimensions
        slices = image_slices(img, slice_height, slice_width)
        # iterate over the slices
        for i, gt_slice in enumerate(slices):
            i_string = str(i)
            i_string = i_string.rjust(4, '0')
            new_directory_name = directory_name+"_"+i_string+"/"
            os.makedirs(base_destination_directory+new_directory_name)
            gt_slice_name = "GT_"+base_file_name+"_"+i_string+extension_file_name
            noisy_slice_name = noisy_image_prefix+base_file_name+"_"+i_string+extension_file_name
            noisy_slice = noise_function(gt_slice)
            cv2.imwrite(base_destination_directory+new_directory_name+gt_slice_name, gt_slice)
            cv2.imwrite(base_destination_directory+new_directory_name+noisy_slice_name, noisy_slice)

np.random.seed(256)

original_dataset_path = Path("../SIDD_Small_sRGB_Only/Data")
# new_dataset_path = Path("../SIDD_Small_sliced/Data/")
new_dataset_path_string = "../SIDD_Small_sliced/Data/"
os.makedirs(new_dataset_path_string)
count = 0
for image_directory in original_dataset_path.iterdir():
    if not image_directory.is_dir():
        continue
    count+=1
    new_sliced_and_add_noisy_directory(image_directory, new_dataset_path_string)
    print(f"Finished directory number {count}: {image_directory}")
