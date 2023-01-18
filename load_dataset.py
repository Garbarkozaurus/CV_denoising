import numpy as np
import cv2
from pathlib import Path
from random import shuffle

def load_dataset_directory(path_str: str, ground_truth_name_pattern: str="GT_*.PNG", noisy_name_pattern: str = "GAUSS_*.PNG"):
    directory = Path(path_str)

    ground_truth_image = None
    noisy_image = None
    for file in directory.glob(f"**/{ground_truth_name_pattern}"):
        if not file.is_file():
            continue
        ground_truth_image = cv2.imread(str(file))
        break

    for file in directory.glob(f"**/{noisy_name_pattern}"):
        if not file.is_file():
            continue
        noisy_image = cv2.imread(str(file))
        break

    return ground_truth_image, noisy_image

def load_dataset(base_path_str: str = "../SIDD_Small_sliced/Data", training_examples: int = 10000, test_examples: int = 1250, validataion_examples: int = 1250):
    base_path = Path(base_path_str)
    subdirectory_names = []
    for directory in base_path.iterdir():
        if not directory.is_dir():
            continue
        # saving as strings to save memory
        subdirectory_names.append(str(directory))
    shuffle(subdirectory_names)
    subdirectory_names = subdirectory_names[:training_examples+test_examples+validataion_examples]
    # check type of loaded arrays
    training_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[:training_examples]])
    test_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[training_examples:training_examples+test_examples]])
    validation_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[training_examples:training_examples+test_examples:]])
    training_input, training_output = training_pairs[:, 0, :, :, :], training_pairs[:, 1, :, :, :]
    test_input, test_output = test_pairs[:, 0, :, :, :], test_pairs[:, 1, :, :, :]
    validation_input, validation_output = validation_pairs[:, 0, :, :, :], validation_pairs[:, 1, :, :, :]
    return training_input, training_output, test_input, test_output, validation_input, validation_output