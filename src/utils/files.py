import glob
import os
from typing import List


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

    # Use glob to get all files with the specified extension
    file_paths = glob.glob(os.path.join(directory, "*" + extension))

    # Raise an error if no files are found
    if not file_paths:
        raise FileNotFoundError(
            f"No files with the extension {extension} found in directory {directory}"
        )

    return file_paths
