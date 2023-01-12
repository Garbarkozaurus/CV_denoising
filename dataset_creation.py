import numpy as np
import cv2
from pathlib import Path

def image_slices(image: np.ndarray, slice_height: int, slice_width: int):
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
            slice_list.append(image[top_left_y:top_left_y+slice_height+1, top_left_x:top_left_x+slice_width+1, :])
            top_left_x += slice_width
        top_left_x = 0
        top_left_y += slice_height
    return slice_list

# TODO: fix extreme values occurring on dark pixels (probably has sth to do with uint8 limits)
# look into whether gaussian noise can be added to colour images
# maybe find an alternative approach
def image_with_gaussian_noise(image: np.ndarray, mean: int = 0, std: float = 0.05):
    row,col,ch= image.shape
    gauss = np.random.normal(mean,std,(row,col,ch))
    gauss = (255*np.abs(gauss.reshape(row,col,ch))).astype(np.uint8)

    noisy_image = image + gauss
    return noisy_image

# work in progress - still need to
def process_directory(posix_directory):
    for file in posix_directory.glob("**/GT*.PNG"):
        file_path = str(file)
        img = cv2.imread(file_path)
        # TODO


np.random.seed(256)
# p = Path("../SIDD_Small_sRGB_Only/Data")
# for image_directory in p.iterdir():
#     if not image_directory.is_dir():
#         continue
#     process_directory(image_directory)

test_path = Path("../../../../../Pictures")
test_img = cv2.imread("../../../../../Pictures/clock_test.png")
noisy_clock = image_with_gaussian_noise(test_img)
cv2.imshow("", np.concatenate([noisy_clock, test_img], axis = 1))
ans = cv2.waitKey(0)
while chr(ans) != "q":
    ans = cv2.waitKey(0)

cv2.destroyAllWindows()