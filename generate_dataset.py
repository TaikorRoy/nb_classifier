__author__ = 'roy'

from py_utility.dataset.preprocess import dataset_split
from py_utility import system

if __name__ == "__main__":
    dataset_path = r"data/open_test/no_meiti/dataset.txt"
    training_path = r"data/training_set.txt"
    verification_path = r"data/verification_set.txt"

    dataset = system.get_content_list(dataset_path)
    ratio = 0.8
    training, verification = dataset_split(dataset, ratio)

    sep = "\n"
    training_str = system.to_string(training, sep)
    verification_str = system.to_string(verification, sep)
    system.write_content(training_path, training_str)
    system.write_content(verification_path, verification_str)



