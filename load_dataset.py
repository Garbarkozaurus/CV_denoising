import numpy as np
import cv2
from pathlib import Path
from random import shuffle
from image_standardisation import standardise_image

def load_dataset_directory(path_str: str, ground_truth_name_pattern: str="GT_*.PNG", noisy_name_pattern: str = "GAUSS_*.PNG"):
    directory = Path(path_str)

    ground_truth_image = None
    noisy_image = None
    for file in directory.glob(f"**/{ground_truth_name_pattern}"):
        if not file.is_file():
            continue
        ground_truth_image = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
        break

    for file in directory.glob(f"**/{noisy_name_pattern}"):
        if not file.is_file():
            continue
        noisy_image = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
        break

    return noisy_image, ground_truth_image

def load_dataset(base_path_str: str = "../SIDD_Small_sliced/Data", training_examples: int = 10000, test_examples: int = 1250, validation_examples: int = 1250):
    base_path = Path(base_path_str)
    subdirectory_names = []
    for directory in base_path.iterdir():
        if not directory.is_dir():
            continue
        # saving as strings to save memory
        subdirectory_names.append(str(directory))
    shuffle(subdirectory_names)
    subdirectory_names = subdirectory_names[:training_examples+test_examples+validation_examples]

    training_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[:training_examples]])
    test_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[training_examples:training_examples+test_examples]])
    validation_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[training_examples:training_examples+test_examples:]])
    training_input, training_output = training_pairs[:, 0, :, :, :], training_pairs[:, 1, :, :, :]
    test_input, test_output = test_pairs[:, 0, :, :, :], test_pairs[:, 1, :, :, :]
    validation_input, validation_output = validation_pairs[:, 0, :, :, :], validation_pairs[:, 1, :, :, :]
    return training_input, training_output, test_input, test_output, validation_input, validation_output

def load_standardised_dataset(base_path_str: str = "../SIDD_Small_sliced/Data", training_examples: int = 10000, test_examples: int = 1250, validation_examples: int = 1250):
    base_path = Path(base_path_str)
    subdirectory_names = []
    for directory in base_path.iterdir():
        if not directory.is_dir():
            continue
        # saving as strings to save memory
        subdirectory_names.append(str(directory))
    shuffle(subdirectory_names)
    subdirectory_names = subdirectory_names[:training_examples+test_examples+validation_examples]

    training_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[:training_examples]])
    test_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[training_examples:training_examples+test_examples]])
    validation_pairs = np.array([load_dataset_directory(dir) for dir in subdirectory_names[training_examples:training_examples+test_examples:]])
    standard_training_input = np.array([standardise_image(img) for img in training_pairs[:, 0, :, :, :]])
    standard_training_output =  np.array([standardise_image(img) for img in training_pairs[:, 1, :, :, :]])
    standard_test_input = np.array([standardise_image(img) for img in test_pairs[:, 0, :, :, :]])
    standard_test_output = np.array([standardise_image(img) for img in test_pairs[:, 1, :, :, :]])
    standard_validation_input = np.array([standardise_image(img) for img in validation_pairs[:, 0, :, :, :]])
    standard_validation_output =  np.array([standardise_image(img) for img in validation_pairs[:, 1, :, :, :]])
    return standard_training_input, standard_training_output, standard_test_input, standard_test_output, standard_validation_input, standard_validation_output