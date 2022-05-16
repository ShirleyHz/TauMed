from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from PIL.Image import Image
from builtins import *

from .OverallOperations import *
from .LesionOperations import *
from .ImageUtilities import scan_directory, scan, scan_dataframe, TauMedImage, unzip, unRar, saveUnzipImages

import os
import sys
import random
import uuid
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Pipeline(object):
    # Some class variables we use often
    _probability_error_text = "The probability argument must be between 0 and 1."
    _threshold_error_text = "The value of threshold must be between 0 and 255."
    _valid_formats = ["PNG", "BMP", "GIF", "JPEG", "JPG", "TIF"]
    _legal_filters = ["NEAREST", "BICUBIC", "ANTIALIAS", "BILINEAR"]

    def __init__(self, source_directory=None, zip_file = None, output_directory="output", save_format=None):

        # TODO: Allow a single image to be added when initialising.
        # Initialise some variables for the Pipeline object.
        self.image_counter = 0
        self.augmented_images = []
        self.distinct_dimensions = set()
        self.distinct_formats = set()
        self.save_format = save_format
        self.overall_operations = []
        self.lesion_operations = []
        self.class_labels = []
        self.process_ground_truth_images = False

        if source_directory is not None:
            # source_directory = os.path.abspath(source_directory)
            self._populate(source_directory=source_directory,
                           zip_file=zip_file,
                           output_directory=output_directory,
                           ground_truth_directory=None,
                           ground_truth_output_directory=output_directory)

    def __call__(self, augmented_image):

        return self._execute(augmented_image)

    def _populate(self, source_directory, zip_file, output_directory, ground_truth_directory, ground_truth_output_directory):
        if zip_file.endswith(".zip"):
            unzip(source_directory, zip_file)
        elif zip_file.endswith(".rar"):
            unRar(source_directory, zip_file)
        source_directory = saveUnzipImages(source_directory, zip_file)

        # Check if the source directory for the original images to augment exists at all
        if not os.path.exists(source_directory):
            raise IOError("The source directory you specified does not exist.")

        # If a ground truth directory is being specified we will check if the path exists at all.
        if ground_truth_directory:
            if not os.path.exists(ground_truth_directory):
                raise IOError("The ground truth source directory you specified does not exist.")

        # Get absolute path for output
        root_dir = os.getcwd()
        abs_output_directory = os.path.join(root_dir, output_directory)
        print(abs_output_directory)

        # Scan the directory that user supplied
        self.augmented_images, self.class_labels = scan(source_directory, abs_output_directory)

        self._check_images(abs_output_directory)

    def _check_images(self, abs_output_directory):

        # Make output directory/directories
        if len(set(self.class_labels)) <= 1:
            if not os.path.exists(abs_output_directory):
                try:
                    os.makedirs(abs_output_directory)
                except IOError:
                    print("Insufficient rights to read or write output directory (%s)"
                          % abs_output_directory)
        else:
            for class_label in self.class_labels:
                if not os.path.exists(os.path.join(abs_output_directory, str(class_label[0]))):
                    try:
                        os.makedirs(os.path.join(abs_output_directory, str(class_label[0])))
                    except IOError:
                        print("Insufficient rights to read or write output directory (%s)"
                              % abs_output_directory)
        for augmented_image in self.augmented_images:
            try:
                with Image.open(augmented_image.image_path) as opened_image:
                    # The image is too small
                    if opened_image.width <= 1 and opened_image.height <= 1:
                        self.augmented_images.remove(augmented_image)
                    if opened_image is None:
                        self.augmented_images.remove(augmented_image)
                    self.distinct_dimensions.add(opened_image.size)
                    self.distinct_formats.add(opened_image.format)
            except IOError as e:
                print("There is a problem with image %s in your source directory: %s"
                      % (augmented_image.image_path, e.message))
                self.augmented_images.remove(augmented_image)


        sys.stdout.write("Initialised with %s image(s) found.\n" % len(self.augmented_images))
        sys.stdout.write("Output directory set to %s." % abs_output_directory)

    def _execute(self, image, save_to_disk=True, multi_threaded=True):

        images = []

        if image.image_path is not None:
            images.append(Image.open(image.image_path).convert('RGB'))

        if image.pil_images is not None:
            images.append(image.pil_images)

        if image.ground_truth is not None:
            if isinstance(image.ground_truth, list):
                for image in image.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(image.ground_truth))

        for operation in self.overall_operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        for operation in self.lesion_operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        # TEMP FOR TESTING
        # save_to_disk = False

        if save_to_disk:
            file_name = str(uuid.uuid4())
            try:
                for i in range(len(images)):
                    if i == 0:
                        save_name = image.class_label \
                                    + "_original_" \
                                    + os.path.basename(image.image_path) \
                                    + "_" \
                                    + file_name \
                                    + "." \
                                    + (self.save_format if self.save_format else image.file_format)

                        images[i].save(os.path.join(image.output_directory, save_name))

                    else:
                        save_name = "_groundtruth_(" \
                                    + str(i) \
                                    + ")_" \
                                    + image.class_label \
                                    + "_" \
                                    + os.path.basename(image.image_path) \
                                    + "_" \
                                    + file_name \
                                    + "." \
                                    + (self.save_format if self.save_format else image.file_format)

                        images[i].save(os.path.join(image.output_directory, save_name))

            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")

        return images[0]

    def _execute_with_array(self, image):

        pil_image = [Image.fromarray(image)]

        for operation in self.overall_operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                pil_image = operation.perform_operation(pil_image)

        for operation in self.lesion_operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                pil_image = operation.perform_operation(pil_image)

        numpy_array = np.asarray(pil_image[0])

        return numpy_array

    def set_save_format(self, save_format):

        if save_format == "auto":
            self.save_format = None
        else:
            self.save_format = save_format

    def sample(self, n, multi_threaded=True):

        if len(self.augmented_images) == 0:
            raise IndexError("There are no images in the pipeline. "
                             "Add a directory using add_directory(), "
                             "pointing it to a directory containing images.")

        if len(self.overall_operations or self.lesion_operations) == 0:
            raise IndexError("There are no operations associated with this pipeline.")

        if n == 0:
            augmented_images = self.augmented_images
        else:
            augmented_images = [random.choice(self.augmented_images) for _ in range(n)]

        # for image in augmented_images:
        #     file_name = image.image_file_name()

        if multi_threaded:
            # TODO: Restore the functionality (appearance of progress bar) from the pre-multi-thread code above.
            with tqdm(total=len(augmented_images), desc="Executing Pipeline", unit=" Samples") as progress_bar:
                with ThreadPoolExecutor(max_workers=None) as executor:
                    for result in executor.map(self, augmented_images):
                        progress_bar.set_description("Processing %s" % result)
                        progress_bar.update(1)
        else:
            with tqdm(total=len(augmented_images), desc="Executing Pipeline", unit=" Samples") as progress_bar:
                for augmentor_image in augmented_images:
                    self._execute(augmentor_image)
                    progress_bar.set_description("Processing %s" % os.path.basename(augmentor_image.image_path))
                    progress_bar.update(1)

        # This does not work as it did in the pre-multi-threading code above for some reason.
        # progress_bar.close()

    def process(self):

        self.sample(0, multi_threaded=True)

        return None

    def sample_with_array(self, image_array, save_to_disk=False):

        a = TauMedImage(image_path=None, output_directory=None)
        a.image_PIL = Image.fromarray(image_array)

        return self._execute(a, save_to_disk)

    @staticmethod
    def categorical_labels(numerical_labels):

        # class_labels_np = np.array([x.class_label_int for x in numerical_labels])
        class_labels_np = np.array(numerical_labels)
        one_hot_encoding = np.zeros((class_labels_np.size, class_labels_np.max() + 1))
        one_hot_encoding[np.arange(class_labels_np.size), class_labels_np] = 1
        one_hot_encoding = one_hot_encoding.astype(np.uint)

        return one_hot_encoding

    def image_generator(self):

        warnings.warn("This function has been deprecated.", DeprecationWarning)

        while True:
            im_index = random.randint(0, len(self.augmented_images) - 1)  # Fix for issue 52.
            yield self._execute(self.augmented_images[im_index], save_to_disk=False)

    def generator_threading_tests(self, batch_size):

        while True:

            return_results = []

            augmented_images = [random.choice(self.augmented_images) for _ in range(batch_size)]

            with ThreadPoolExecutor(max_workers=None) as executor:
                for result in executor.map(self, augmented_images):
                    return_results.append(result)

            yield return_results

    def generator_threading_tests_with_matrix_data(self, images, label):

        self.augmented_images = [TauMedImage(image_path=None, output_directory=None, pil_images=x, label=y)
                                 for x, y in zip(images, label)]

        return 1

    def add_overall_operation(self, operation):

        if isinstance(operation, OverallOperation):
            self.overall_operations.append(operation)
        else:
            raise TypeError("Must be of type OverallOperation to be added to the pipeline.")

    def add_lesion_operation(self, operation):

        if isinstance(operation, LesionOperations):
            self.lesion_operations.append(operation)
        else:
            raise TypeError("Must be of type LesionOperation to be added to the pipeline.")

    def remove_overall_operation(self, operation_index=-1):

        # Python's own List exceptions can handle erroneous user input.
        self.overall_operations.pop(operation_index)

    def remove_lesion_operation(self, operation_index=-1):

        # Python's own List exceptions can handle erroneous user input.
        self.lesion_operations.pop(operation_index)

    def add_further_directory(self, new_source_directory, new_output_directory="output"):

        if not os.path.exists(new_source_directory):
            raise IOError("The path does not appear to exist.")

        self._populate(source_directory=new_source_directory,
                       output_directory=new_output_directory,
                       ground_truth_directory=None,
                       ground_truth_output_directory=new_output_directory)

    def status(self):

        # TODO: Return this as a dictionary of some kind and print from the dict if in console
        print("Overall Operations: %s" % len(self.overall_operations))

        if len(self.overall_operations) != 0:
            operation_index = 0
            for operation in self.overall_operations:
                print("\t%s: %s (" % (operation_index, operation), end="")
                for operation_attribute, operation_value in operation.__dict__.items():
                    print("%s=%s " % (operation_attribute, operation_value), end="")
                print(")")
                operation_index += 1

        print("Images: %s" % len(self.augmented_images))

        # TODO: find a better way that doesn't need to iterate over every image
        # TODO: get rid of this label_pair property as nowhere else uses it
        # Check if we have any labels before printing label information.
        label_count = 0
        for image in self.augmented_images:
            if image.label_pair is not None:
                label_count += 1

        if label_count != 0:
            label_pairs = sorted(set([x.label_pair for x in self.augmented_images]))

            print("Classes: %s" % len(label_pairs))

            for label_pair in label_pairs:
                print("\tClass index: %s Class label: %s " % (label_pair[0], label_pair[1]))

        if len(self.augmented_images) != 0:
            print("Dimensions: %s" % len(self.distinct_dimensions))
            for distinct_dimension in self.distinct_dimensions:
                print("\tWidth: %s Height: %s" % (distinct_dimension[0], distinct_dimension[1]))
            print("Formats: %s" % len(self.distinct_formats))
            for distinct_format in self.distinct_formats:
                print("\t %s" % distinct_format)

        print("\nYou can remove operations using the appropriate index and the remove_operation(index) function.")

    @staticmethod
    def set_seed(seed):

        random.seed(seed)

    def rotate(self, probability, max_left_angle, max_right_angle):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        if not 0 <= max_left_angle <= 25:
            raise ValueError("The max_left_rotation argument must be between 0 and 25.")
        if not 0 <= max_right_angle <= 25:
            raise ValueError("The max_right_rotation argument must be between 0 and 25.")
        else:
            self.add_overall_operation(Rotate(probability=probability, maxLeftAngle=ceil(max_left_angle),
                                              maxRightAngle=ceil(max_right_angle)))

    def flip(self, probability, direction):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(Flip(probability=probability, direction=direction))

    def flip_top_bottom(self, probability):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(Flip(probability=probability, direction="TOP_BOTTOM"))

    def flip_left_right(self, probability):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(Flip(probability=probability, direction="LEFT_RIGHT"))

    def flip_random(self, probability):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(Flip(probability=probability, direction="RANDOM"))

    def random_distortion(self, probability, grid_width, grid_height, magnitude):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(Distort(probability=probability, grid_width=grid_width,
                                               grid_height=grid_height, magnitude=magnitude))

    def gaussian_distortion(self, probability, grid_width, grid_height, magnitude, corner, method, mex=0.5, mey=0.5,
                            sdx=0.05, sdy=0.05):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(GaussianDistortion(probability=probability, grid_width=grid_width,
                                                          grid_height=grid_height,
                                                          magnitude=magnitude, corner=corner,
                                                          method=method, mex=mex,
                                                          mey=mey, sdx=sdx, sdy=sdy))

    def optical_distortion(self,probability, distort_limit=0.05, shift_limit=0.05):
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(OpticalDistortion(probability=probability, distort_limit=distort_limit,
                                                         shift_limit=shift_limit))

    def zoom(self, probability, min_factor, max_factor):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif min_factor <= 0:
            raise ValueError("The min_factor argument must be greater than 0.")
        elif not min_factor <= max_factor:
            raise ValueError("The max_factor must be bigger min_factor.")
        else:
            self.add_overall_operation(Zoom(probability=probability, min_factor=min_factor, max_factor=max_factor))

    def crop_centre(self, probability, percentage_area, randomise_percentage_area=False):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0.1 <= percentage_area < 1:
            raise ValueError("The percentage_area argument must be greater than 0.1 and less than 1.")
        elif not isinstance(randomise_percentage_area, bool):
            raise ValueError("The randomise_percentage_area argument must be True or False.")
        else:
            self.add_overall_operation(
                CropPercentage(probability=probability, percentage_area=percentage_area, centre=True,
                               randomise_percentage_area=randomise_percentage_area))

    def crop_random(self, probability, percentage_area, randomise_percentage_area=False):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0.1 <= percentage_area < 1:
            raise ValueError("The percentage_area argument must be greater than 0.1 and less than 1.")
        elif not isinstance(randomise_percentage_area, bool):
            raise ValueError("The randomise_percentage_area argument must be True or False.")
        else:
            self.add_overall_operation(
                CropPercentage(probability=probability, percentage_area=percentage_area, centre=False,
                               randomise_percentage_area=randomise_percentage_area))

    def histogram_equalisation(self, probability=1.0):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_overall_operation(HistogramEqualisation(probability=probability))

    def resize(self, probability, width, height, resample_filter="BICUBIC"):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not width > 1:
            raise ValueError("Width must be greater than 1.")
        elif not height > 1:
            raise ValueError("Height must be greater than 1.")
        elif resample_filter not in Pipeline._legal_filters:
            raise ValueError("The save_filter argument must be one of %s." % Pipeline._legal_filters)
        else:
            self.add_overall_operation(
                Resize(probability=probability, width=width, height=height, resample_filter=resample_filter))

    def skew_left_right(self, probability, magnitude=1):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_overall_operation(Skew(probability=probability, skew_type="TILT_LEFT_RIGHT", magnitude=magnitude))

    def skew_top_bottom(self, probability, magnitude=1):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_overall_operation(Skew(probability=probability,
                                            skew_type="TILT_TOP_BOTTOM",
                                            magnitude=magnitude))

    def skew_tilt(self, probability, magnitude=1):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_overall_operation(Skew(probability=probability,
                                            skew_type="TILT",
                                            magnitude=magnitude))

    def skew_corner(self, probability, magnitude=1):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_overall_operation(Skew(probability=probability,
                                            skew_type="CORNER",
                                            magnitude=magnitude))

    def skew(self, probability, skew_type, magnitude=1):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_overall_operation(Skew(probability=probability,
                                            skew_type=skew_type,
                                            magnitude=magnitude))

    def shear(self, probability, max_shear_left, max_shear_right):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < max_shear_left <= 25:
            raise ValueError("The max_shear_left argument must be between 0 and 25.")
        elif not 0 < max_shear_right <= 25:
            raise ValueError("The max_shear_right argument must be between 0 and 25.")
        else:
            self.add_overall_operation(Shear(probability=probability,
                                             max_shear_left=max_shear_left,
                                             max_shear_right=max_shear_right))

    def random_brightness(self, probability, min_factor, max_factor):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 <= min_factor <= max_factor:
            raise ValueError("The min_factor must be between 0 and max_factor.")
        elif not min_factor <= max_factor:
            raise ValueError("The max_factor must be bigger min_factor.")
        else:
            self.add_overall_operation(
                RandomBrightness(probability=probability, min_factor=min_factor, max_factor=max_factor))

    def random_erasing(self, probability, rectangle_area):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0.1 < rectangle_area <= 1:
            raise ValueError("The rectangle_area must be between 0.1 and 1.")
        else:
            self.add_overall_operation(RandomErasing(probability=probability, rectangle_area=rectangle_area))

    def lesion_erasing(self, probability):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_lesion_operation(LesionErasing(probability=probability))

    def lesion_random_brightness(self, probability, min_factor, max_factor):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 <= min_factor <= max_factor:
            raise ValueError("The min_factor must be between 0 and max_factor.")
        elif not min_factor <= max_factor:
            raise ValueError("The max_factor must be bigger min_factor.")
        else:
            self.add_lesion_operation(
                LesionRandomBrightness(probability=probability, min_factor=min_factor, max_factor=max_factor))

    def lesion_paste(self, probability, lesion_num=1):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif lesion_num <= 0:
            raise ValueError("The lesion num must be bigger than 0.")
        else:
            self.add_lesion_operation(LesionPaste(probability=probability, lesion_num=lesion_num))

    def lesion_expansion(self, probability, magnitude=1.2):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_lesion_operation(LesionExpansion(probability=probability, magnitude=magnitude))

    def add_nonlesion_noises(self, probability):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_lesion_operation(AddNonLesionNoises(probability=probability))

    def nonlesion_contrast(self, probability, factor):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not factor >= 0:
            raise ValueError("The factor must be bigger than 0.")
        else:
            self.add_lesion_operation(NonLesionContrast(probability=probability, factor=factor))

    def nonlesion_sharpness(self, probability, factor):

        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not factor >= 0:
            raise ValueError("The factor must be bigger than 0.")
        else:
            self.add_lesion_operation(NonLesionSharpness(probability=probability, factor=factor))

    def ground_truth(self, ground_truth_directory):

        num_of_ground_truth_images_added = 0

        # Progress bar
        progress_bar = tqdm(total=len(self.augmented_images), desc="Processing", unit=' Images', leave=False)

        if len(self.class_labels) == 1:
            for augmentor_image_idx in range(len(self.augmented_images)):
                ground_truth_image = os.path.join(ground_truth_directory,
                                                  self.augmented_images[augmentor_image_idx].image_file_name)
                if os.path.isfile(ground_truth_image):
                    self.augmented_images[augmentor_image_idx].ground_truth = ground_truth_image
                    num_of_ground_truth_images_added += 1
        else:
            for i in range(len(self.class_labels)):
                for augmentor_image_idx in range(len(self.augmented_images)):
                    ground_truth_image = os.path.join(ground_truth_directory,
                                                      self.augmented_images[augmentor_image_idx].class_label,
                                                      self.augmented_images[augmentor_image_idx].image_file_name)
                    if os.path.isfile(ground_truth_image):
                        if self.augmented_images[augmentor_image_idx].class_label == self.class_labels[i][0]:
                            # Check files are the same size. There may be a better way to do this.
                            original_image_dimensions = \
                                Image.open(self.augmented_images[augmentor_image_idx].image_path).size
                            ground_image_dimensions = Image.open(ground_truth_image).size
                            if original_image_dimensions == ground_image_dimensions:
                                self.augmented_images[augmentor_image_idx].ground_truth = ground_truth_image
                                num_of_ground_truth_images_added += 1
                                progress_bar.update(1)

        progress_bar.close()

        # May not be required after all, check later.
        if num_of_ground_truth_images_added != 0:
            self.process_ground_truth_images = True

        print("%s ground truth image(s) found." % num_of_ground_truth_images_added)

    def get_ground_truth_paths(self):

        paths = []

        for augmented_image in self.augmented_images:
            print("Image path: %s\nGround truth path: %s\n---\n" % (
            augmented_image.image_path, augmented_image.ground_truth))
            paths.append((augmented_image.image_path, augmented_image.ground_truth))

        return paths
