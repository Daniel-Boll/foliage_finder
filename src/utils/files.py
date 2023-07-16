import glob
import os
import random
from typing import List


def get_balanced_files(directory: str, extension: str, num_files: int) -> list:
    """
    Returns a balanced number of files from each sub-directory in the given directory.

    Parameters:
    directory (str): The main directory to search in.
    extension (str): The file extension to look for.
    num_files (int): The number of files to take from each sub-directory.

    Returns:
    list: A list of file paths across all sub-directories.

    Raises:
    FileNotFoundError: If no files with the specified extension are found in a sub-directory.
    """
    # Make sure the extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension

    all_files = []

    for class_dir in os.listdir(directory):
        # Use glob to get all files with the specified extension in the sub-directory
        file_paths = glob.glob(
            os.path.join(directory, class_dir, "*" + extension))

        # Raise an error if no files are found
        if not file_paths:
            raise FileNotFoundError(
                f"No files with the extension {extension} found in directory {os.path.join(directory, class_dir)}"
            )

        # Check if there are enough files in the directory
        if len(file_paths) < num_files:
            raise ValueError(
                f"Not enough files in directory {os.path.join(directory, class_dir)}. Requested {num_files}, but found {len(file_paths)}."
            )

        # Randomly select num_files from the directory and add to the all_files list
        all_files.extend(random.sample(file_paths, num_files))

    return all_files


def get_files_from_dir(directory: str, extension: str) -> List[str]:
    """
    Returns all files in the given directory with the given extension.

    Parameters:
    directory (str): The directory to search in.
    extension (str): The file extension to look for.

    Returns:
    List[str]: A list of file paths.

    Raises:
    FileNotFoundError: If no files with the specified extension are found.
    """
    # Make sure the extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension

    # Use glob to get all files with the specified extension, and set recursive to True
    file_paths = glob.glob(os.path.join(directory, "**", "*" + extension),
                           recursive=True)

    # Raise an error if no files are found
    if not file_paths:
        raise FileNotFoundError(
            f"No files with the extension {extension} found in directory {directory}"
        )

    return file_paths
