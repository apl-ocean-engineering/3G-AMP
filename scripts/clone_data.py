import os
import re
import sys

from shutil import copy2

PATTERN_LVL_1 = r'20\d{2}_\d{2}_\d{2}' # regular expression for directory names on sub level 1
PATTERN_LVL_2 = r'20\d{2}_\d{2}_\d{2} \d{2}_\d{2}_\d{2}' # regular expression for directory names on sub level 2
PATTERN_LVL_3 = r'Manta \d' # regular expression for directory names on sub level 3
PATTERN_LVL_4 = r'20\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.\d+\.jpg' # regular expression for directory names on sub level 4


def getSubDirectoriesWithPattern(root_directory, pattern):
    """
    Given a path to the root directory, this function returns a list
    of immediate sub-directories with names which match the pattern.

    Input:
    root_directory (str): absolute path to the root directory
    pattern (str): regular expression to filter directory names

    Output:
    subdirs (list): List of absolute paths to sub-directories
    """
    subdirs = [f.path for f in os.scandir(root_directory) if f.is_dir() and re.match(pattern=pattern, string=f.name)]
    return subdirs


def getFilesWithPattern(root_directory, pattern):
    """
    Given a path to the root directory, this function returns a list
    of files inside the directory with names which match the pattern.

    Input:
    root_directory (str): absolute path to the root directory
    pattern (str): regular expression to filter file names

    Output:
    files (list): List of absolute paths to files
    """
    files = [f.path for f in os.scandir(root_directory) if f.is_file() and re.match(pattern=pattern, string=f.name)]
    return files


def flattenAndCopyFiles(orig_path, new_path, patterns):
    """
    Copies files from the subdirectories of the orig_path to the
    new_path directory. The level and choice of subdirectories to
    penetrate is specified by the patterns variable which contains
    a list of regular expressions.

    Input:
    orig_path (str): absolute path to the root directory containing the files
                     and folders
    new_path (str):  absolute path to the new directory to dump the files
    patterns (list): regular expressions to filter each level of directory and
                     file names
    """
    directory_paths = [orig_path]  # list to accumulate path to intermediate directories
    directory_levels = len(patterns) - 1  # Total levels of directories to penetrate

    # Get a list of all final-level directories to look for files in
    for level in range(directory_levels):
        new_directories = []

        for directory in directory_paths:
            new_directories.extend(getSubDirectoriesWithPattern(root_directory=directory,
                                                                pattern=patterns[level]))

        directory_paths = new_directories

    # Move files from each directory to the new path
    for directory in directory_paths:
        filepaths = getFilesWithPattern(root_directory=directory,
                                        pattern=patterns[-1])
        for file in filepaths:
            copy2(file, new_path)

    return


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: python clone_data.py [SOURCE] [DESTINATION]")
        sys.exit()

    BASE_INPUT_PATH = sys.argv[1]  # path to directory containing the folders of images
    BASE_OUTPUT_PATH = sys.argv[2]  # path to the directory to dump the images

    patterns = [PATTERN_LVL_1, PATTERN_LVL_2, PATTERN_LVL_3, PATTERN_LVL_4]

    flattenAndCopyFiles(orig_path=BASE_INPUT_PATH, new_path=BASE_OUTPUT_PATH, patterns=patterns)
