# ImageUtilities.py
# Author: Marcus D. Bloice <https://github.com/mdbloice> and contributors
# Licensed under the terms of the MIT Licence.
"""
The ImageUtilities module provides a number of helper functions, as well as
the main :class:`~Augmentor.ImageUtilities.AugmentorImage` class, that is used
throughout the package as a container class for images to be augmented.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import os
import glob
import numbers
import random
import warnings
import numpy as np
import zipfile
from unrar import rarfile
from PIL import Image
import shutil


class TauMedImage(object):
    """
    Wrapper class containing paths to images, as well as a number of other
    parameters, that are used by the Pipeline and Operation modules to perform
    augmentation.
    Each image that is found by Augmentor during the initialisation of a
    Pipeline object is contained with a new AugmentorImage object.
    """
    def __init__(self,
                 image_path,
                 output_directory,
                 pil_images=None,
                 array_images=None,
                 path_images=None,
                 class_label_int=None):
        """
        To initialise an AugmentorImage object for any image, the image's
        file path is required, as well as that image's output directory,
        which defines where any augmented images are stored.
        :param image_path: The full path to an image.
        :param output_directory: The directory where augmented images for this
         image should be saved.
        """

        # Could really think about initialising AugmentorImage member
        # variables here and and only here during init. Then remove all
        # setters below so that they cannot be altered later.

        # Call the setters from parameters that are required.
        self._image_path = image_path
        self._output_directory = output_directory

        self._ground_truth = None

        self._image_paths = None
        self._image_arrays = None
        self._pil_images = None
        # self._lesion_position = None

        self._file_format = None
        self._class_label = None
        self._class_label_int = None
        self._label = None
        self._label_pair = None
        self._categorical_label = None

        if pil_images is not None:
            self._pil_images = pil_images

        if array_images is not None:
            self._array_images = array_images

        if path_images is not None:
            self._path_images = path_images

        if class_label_int is not None:
            self._class_label_int = class_label_int

        # if lesion_position is not None:
        #     self._lesion_position = lesion_position

    def __str__(self):
        return """
        Image path: %s
        Ground truth path: %s
        File format (inferred from extension): %s
        Class label: %s
        Numerical class label (auto assigned): %s
        """ % (self._image_path, self._ground_truth, self._file_format, self._class_label, self._class_label_int)

    @property
    def pil_images(self):
        return self._pil_images

    @pil_images.setter
    def pil_images(self, value):
        self._pil_images = value

    @property
    def image_arrays(self):
        return self._image_arrays

    @image_arrays.setter
    def image_arrays(self, value):
        self._image_arrays = value

    @property
    def class_label_int(self):
        return self._class_label_int

    @class_label_int.setter
    def class_label_int(self, value):
        self._class_label_int = value

    @property
    def output_directory(self):
        return self._output_directory

    @output_directory.setter
    def output_directory(self, value):
        self._output_directory = value

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        self._image_path = value

    @property
    def image_file_name(self):
        return os.path.basename(self._image_path)

    @property
    def class_label(self):
        return self._class_label

    @class_label.setter
    def class_label(self, value):
        self._class_label = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def categorical_label(self):
        return self._categorical_label

    @categorical_label.setter
    def categorical_label(self, value):
        self._categorical_label = value

    @property
    def ground_truth(self):
        """
        The :attr:`ground_truth` property contains an absolute path to the
        ground truth file for an image.
        :getter: Returns this image's ground truth file path.
        :setter: Sets this image's ground truth file path.
        :type: String
        """
        return self._ground_truth

    @ground_truth.setter
    def ground_truth(self, value):
        if os.path.isfile(value):
            self._ground_truth = value

    @property
    def label_pair(self):
        return self._class_label_int, self._class_label

    @property
    def file_format(self):
        return self._file_format

    @file_format.setter
    def file_format(self, value):
        self._file_format = value

    # @property
    # def lesion_position(self):
    #     """
    #     The coordinates of the lesion area
    #     :return: list[x1,y1,x2,y2]
    #     """
    #     return self._lesion_position
    #
    # @lesion_position.setter
    # def lesion_position(self,value):
    #     self._lesion_position = value


def parse_user_parameter(user_param):

    if isinstance(user_param, numbers.Real):
        return user_param
    elif isinstance(user_param, tuple):
        return random.sample(user_param, 1)[0]
    elif isinstance(user_param, list):
        return random.choice(np.arange(*user_param))


def extract_paths_and_extensions(image_path):

    file_name, extension = os.path.splitext(image_path)
    root_path = os.path.dirname(image_path)

    return file_name, extension, root_path


def unzip(path, zfile):
    file_path = path + os.sep + zfile
    desdir1 = path + os.sep + zfile[:zfile.index('.zip')]
    srcfile = zipfile.ZipFile(file_path)
    for filename in srcfile.namelist():
        filename = filename.encode('utf-8').decode('utf-8')
        srcfile.extract(filename, desdir1)
        if filename.endswith('.zip'):
            # if zipfile.is_zipfile(filename):
            path = desdir1
            zfile = filename
            unzip(path, zfile)
        if filename.endswith('.rar'):
            path = desdir1
            rfile = filename
            unRar(path, rfile)

def unRar(path, rfile):
    file_path = path + os.sep + rfile
    desdir1 = path + os.sep + rfile[:rfile.index('.rar')]
    rf = rarfile.RarFile(file_path)
    for filename in rf.namelist():
        # filename = filename
        rf.extract(filename, desdir1)
        if filename.endswith('.zip'):
            # if zipfile.is_zipfile(filename):
            path = desdir1
            zfile = filename
            unzip(path, zfile)
        if filename.endswith('.rar'):
            path = desdir1
            rfile = filename
            unRar(path, rfile)

def saveUnzipImages(path, zfile):
    # global new_path
    global desdir
    if zfile.endswith('.zip'):
        desdir = path + os.sep + zfile[:zfile.index('.zip')]
    elif zfile.endswith('.rar'):
        desdir = path + os.sep + zfile[:zfile.index('.rar')]
    savePath = path + os.sep + 'images'
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    renameDir(desdir)

    for root_path, dir_names, file_names in os.walk(desdir):
        for dir in dir_names:
            if dir == "__MACOSX":
                del_file(root_path + os.sep + dir)
        for fn in file_names:
            dir_path = os.path.join(root_path + os.sep + fn)
            file_ext = os.path.splitext(dir_path)[1]
            if file_ext in [".png", ".bmp", ".gif", ".jpeg", ".jpg", ".tif"]:
                try:
                    img = Image.open(dir_path)
                    if img.width <= 1 and img.height <= 1:
                        continue
                    if file_ext == ".jpg" or file_ext == ".tif":
                        img.save(savePath + os.sep + fn)
                    else:
                        img.save(savePath + os.sep + fn, file_ext[1:])
                except IOError:
                    print("Cannot read the image: " + dir_path)

    del_file(desdir)
    shutil.rmtree(desdir)
    return savePath


def renameDir(desdir_tmp):
    if desdir_tmp[-1] != '/':
        desdir_tmp = desdir_tmp + '/'
    files = os.listdir(desdir_tmp)
    for file in files:
        old_name = os.path.join(desdir_tmp, file)
        try:
            filename = file.encode('cp437').decode('gbk')  # 解决中文乱码的问题
        except UnicodeEncodeError:
            filename = file.encode('utf-8').decode('utf-8')
        new_name = os.path.join(desdir_tmp, filename)
        os.rename(old_name, new_name)
        if os.path.isdir(new_name):
            renameDir(new_name)

def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def scan(source_directory, output_directory):

    abs_output_directory = os.path.abspath(output_directory)
    files_and_directories = glob.glob(os.path.join(os.path.abspath(source_directory), '*'))

    directory_count = 0
    directories = []

    class_labels = []

    for f in files_and_directories:
        if os.path.isdir(f):
            if f != abs_output_directory:
                directories.append(f)
                directory_count += 1

    directories = sorted(directories)
    label_counter = 0

    if directory_count == 0:

        augmented_images = []
        # This was wrong
        # parent_directory_name = os.path.basename(os.path.abspath(os.path.join(source_directory, os.pardir)))
        parent_directory_name = os.path.basename(os.path.abspath(source_directory))
        for image_path in scan_directory(source_directory):
            a = TauMedImage(image_path=image_path, output_directory=abs_output_directory)
            a.class_label = parent_directory_name
            a.class_label_int = label_counter
            a.categorical_label = [label_counter]
            a.file_format = os.path.splitext(image_path)[1].split(".")[1]
            # filename = os.path.split(image_path)[1]
            # a.lesion_position = patient_df.loc[patient_df['File_name'] == filename]['bbox']
            augmented_images.append(a)

        class_labels.append((label_counter, parent_directory_name))

        return augmented_images, class_labels

    elif directory_count != 0:
        augmented_images = []
        # patient_df = get_lesion_info()
        for d in directories:
            output_directory = os.path.join(abs_output_directory, os.path.split(d)[1])
            for image_path in scan_directory(d):
                categorical_label = np.zeros(directory_count, dtype=np.uint32)
                a = TauMedImage(image_path=image_path, output_directory=output_directory)
                a.class_label = os.path.split(d)[1]
                a.class_label_int = label_counter
                categorical_label[label_counter] = 1  # Set to 1 with the index of the current class.
                a.categorical_label = categorical_label
                a.file_format = os.path.splitext(image_path)[1].split(".")[1]
                # filename = os.path.split(image_path)[1]
                # a.lesion_position = patient_df.loc[patient_df['File_name'] == filename]['bbox']
                augmented_images.append(a)
            class_labels.append((os.path.split(d)[1], label_counter))
            label_counter += 1

        return augmented_images, class_labels


def scan_dataframe(source_dataframe, image_col, category_col, output_directory):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Pandas is required to use the scan_dataframe function!\nrun pip install pandas and try again')

    # ensure column is categorical
    cat_col_series = pd.Categorical(source_dataframe[category_col])
    abs_output_directory = os.path.abspath(output_directory)
    class_labels = list(enumerate(cat_col_series.categories))

    augmented_images = []

    for image_path, cat_name, cat_id in zip(source_dataframe[image_col].values,
                                            cat_col_series.get_values(),
                                            cat_col_series.codes):

        a = TauMedImage(image_path=image_path, output_directory=abs_output_directory)
        a.class_label = cat_name
        a.class_label_int = cat_id
        categorical_label = np.zeros(len(class_labels), dtype=np.uint32)
        categorical_label[cat_id] = 1
        a.categorical_label = categorical_label
        a.file_format = os.path.splitext(image_path)[1].split(".")[1]
        augmented_images.append(a)

    return augmented_images, class_labels


def scan_directory(source_directory):

    # TODO: GIFs are highly problematic. It may make sense to drop GIF support.
    file_types = ['*.jpg', '*.bmp', '*.jpeg', '*.gif', '*.img', '*.png', '*.tiff', '*.tif']

    list_of_files = []

    if os.name == "nt":
        for file_type in file_types:
            list_of_files.extend(glob.glob(os.path.join(os.path.abspath(source_directory), file_type)))
    else:
        file_types.extend([str.upper(str(x)) for x in file_types])
        for file_type in file_types:
            list_of_files.extend(glob.glob(os.path.join(os.path.abspath(source_directory), file_type)))

    return list_of_files


def scan_directory_with_classes(source_directory):
    warnings.warn("The scan_directory_with_classes() function has been deprecated.", DeprecationWarning)
    l = glob.glob(os.path.join(source_directory, '*'))

    directories = []

    for f in l:
        if os.path.isdir(f):
            directories.append(f)

    list_of_files = {}

    for d in directories:
        list_of_files_current_folder = scan_directory(d)
        list_of_files[os.path.split(d)[1]] = list_of_files_current_folder

    return list_of_files

# def get_lesion_info():
#     patient_df = getDeepLesionSamples()
#     return patient_df
