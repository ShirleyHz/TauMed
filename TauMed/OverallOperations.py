import math
from math import floor, ceil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2


class OverallOperation(object):
    """
    对于医疗图像的整体扩增处理操作
    """

    def __init__(self, probability):
        self.probability = probability

    def __str__(self):
        return self.__class__.__name__

    def perform_operation(self, images):
        raise RuntimeError("Illegal call to base class.")


class Rotate(OverallOperation):
    """
    旋转操作
    """

    def __init__(self, probability, maxLeftAngle, maxRightAngle):
        OverallOperation.__init__(self, probability)
        self.maxLeftAngle = -abs(maxLeftAngle)
        self.maxRightAngle = abs(maxRightAngle)

    def perform_operation(self, images):
        random_left_angle = random.randint(self.maxLeftAngle, 0)
        random_right_angle = random.randint(0, self.maxRightAngle)
        rotate_direction = random.randint(0, 1)

        rotation_angle = 0
        # left
        if rotate_direction == 0:
            rotation_angle = random_left_angle
        if rotate_direction == 1:
            rotation_angle = random_right_angle

        def do(image):
            x = image.size[0]
            y = image.size[1]

            image = image.rotate(rotation_angle, expand = True, resample = Image.BICUBIC)

            X = image.size[0]
            Y = image.size[1]

            angle_a = abs(rotation_angle)
            angle_b = 90 - angle_a

            angle_a_rad = math.radians(angle_a)
            angle_b_rad = math.radians(angle_b)

            angle_a_sin = math.sin(angle_a_rad)
            angle_b_sin = math.sin(angle_b_rad)

            E = angle_a_sin / angle_b_sin * (Y - X * (angle_a_sin / angle_b_sin))
            E = E / 1 - (angle_a_sin ** 2 / angle_b_sin ** 2)
            B = X - E
            A = (angle_a_sin / angle_b_sin) * B

            image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

            return image.resize((x, y), resample=Image.BICUBIC)


        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Flip(OverallOperation):
    """
    翻转操作
    :param direction:["LEFT_RIGHT","TOP_BUTTON","RANDOM"]
    """

    def __init__(self, probability, direction):
        OverallOperation.__init__(self, probability)
        self.direction = direction

    def perform_operation(self, images):
        random_direction = random.randint(0, 1)

        def do(image):
            if self.direction == "LEFT_RIGHT":
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.direction == "TOP_BUTTON":
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            elif self.direction == "RANDOM":
                if random_direction == 0:
                    return image.transpose(Image.FLIP_LEFT_RIGHT)
                elif random_direction == 1:
                    return image.transpose(Image.FLIP_TOP_BOTTOM)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class HistogramEqualisation(OverallOperation):
    """
    直方图均衡化
    """

    def __init__(self, probability):
        OverallOperation.__init__(self, probability)

    def perform_operation(self, images):
        def do(image):
            return ImageOps.equalize(image)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RandomBrightness(OverallOperation):
    """
    随机改变图像亮度
    """

    def __init__(self, probability, min_factor, max_factor):
        OverallOperation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        factor = np.random.uniform(self.min_factor, self.max_factor)

        def do(image):
            image_enhancer_brightness = ImageEnhance.Brightness(image)
            return image_enhancer_brightness.enhance(factor)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Skew(OverallOperation):
    """
    This class is used to perform perspective skewing on images. It allows
    for skewing from a total of 12 different perspectives.
    """

    def __init__(self, probability, skew_type, magnitude):

        OverallOperation.__init__(self, probability)
        self.skew_type = skew_type
        self.magnitude = magnitude

    def perform_operation(self, images):
        w, h = images[0].size

        x1 = 0
        x2 = h
        y1 = 0
        y2 = w

        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.skew_type == "RANDOM":
            skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
        else:
            skew = self.skew_type

        if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

            if skew == "TILT":
                skew_direction = random.randint(0, 3)
            elif skew == "TILT_LEFT_RIGHT":
                skew_direction = random.randint(0, 1)
            elif skew == "TILT_TOP_BOTTOM":
                skew_direction = random.randint(2, 3)

            if skew_direction == 0:
                # Left Tilt
                new_plane = [(y1, x1 - skew_amount),  # Top Left
                             (y2, x1),  # Top Right
                             (y2, x2),  # Bottom Right
                             (y1, x2 + skew_amount)]  # Bottom Left
            elif skew_direction == 1:
                # Right Tilt
                new_plane = [(y1, x1),  # Top Left
                             (y2, x1 - skew_amount),  # Top Right
                             (y2, x2 + skew_amount),  # Bottom Right
                             (y1, x2)]  # Bottom Left
            elif skew_direction == 2:
                # Forward Tilt
                new_plane = [(y1 - skew_amount, x1),  # Top Left
                             (y2 + skew_amount, x1),  # Top Right
                             (y2, x2),  # Bottom Right
                             (y1, x2)]  # Bottom Left
            elif skew_direction == 3:
                # Backward Tilt
                new_plane = [(y1, x1),  # Top Left
                             (y2, x1),  # Top Right
                             (y2 + skew_amount, x2),  # Bottom Right
                             (y1 - skew_amount, x2)]  # Bottom Left

        if skew == "CORNER":

            skew_direction = random.randint(0, 7)

            if skew_direction == 0:
                # Skew possibility 0
                new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 1:
                # Skew possibility 1
                new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 2:
                # Skew possibility 2
                new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 3:
                # Skew possibility 3
                new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
            elif skew_direction == 4:
                # Skew possibility 4
                new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
            elif skew_direction == 5:
                # Skew possibility 5
                new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
            elif skew_direction == 6:
                # Skew possibility 6
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
            elif skew_direction == 7:
                # Skew possibility 7
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

        if self.skew_type == "ALL":
            corners = dict()
            corners["top_left"] = (y1 - random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
            corners["top_right"] = (y2 + random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
            corners["bottom_right"] = (y2 + random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))
            corners["bottom_left"] = (y1 - random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))

            new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]

        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(original_plane).reshape(8)

        perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
        perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

        def do(image):
            return image.transform(image.size,
                                   Image.PERSPECTIVE,
                                   perspective_skew_coefficients_matrix,
                                   resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Resize(OverallOperation):
    """
    This class is used to resize images by absolute values passed as parameters.
    """

    def __init__(self, probability, width, height, resample_filter):
        OverallOperation.__init__(self, probability)
        self.width = width
        self.height = height
        self.resample_filter = resample_filter

    def perform_operation(self, images):
        def do(image):
            # TODO: Automatically change this to ANTIALIAS or BICUBIC depending on the size of the file
            return image.resize((self.width, self.height), eval("Image.%s" % self.resample_filter))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class CropPercentage(OverallOperation):
    """
    This class is used to crop images by a percentage of their area.
    """

    def __init__(self, probability, percentage_area, centre, randomise_percentage_area):

        OverallOperation.__init__(self, probability)
        self.percentage_area = percentage_area
        self.centre = centre
        self.randomise_percentage_area = randomise_percentage_area

    def perform_operation(self, images):
        if self.randomise_percentage_area:
            r_percentage_area = round(random.uniform(0.1, self.percentage_area), 2)
        else:
            r_percentage_area = self.percentage_area

        # The images must be of identical size, which is checked by Pipeline.ground_truth().
        w, h = images[0].size

        w_new = int(floor(w * r_percentage_area))  # TODO: Floor might return 0, so we need to check this.
        h_new = int(floor(h * r_percentage_area))

        left_shift = random.randint(0, int((w - w_new)))
        down_shift = random.randint(0, int((h - h_new)))

        def do(image):
            if self.centre:
                image = image.crop(
                    ((w / 2) - (w_new / 2), (h / 2) - (h_new / 2), (w / 2) + (w_new / 2), (h / 2) + (h_new / 2)))
            else:
                image = image.crop((left_shift, down_shift, w_new + left_shift, h_new + down_shift))
            return image.resize((w,h),Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Shear(OverallOperation):
    """
    This class is used to shear images, that is to tilt them in a certain
    direction. Tilting can occur along either the x- or y-axis and in both
    directions (i.e. left or right along the x-axis, up or down along the
    y-axis).
    """

    def __init__(self, probability, max_shear_left, max_shear_right):

        OverallOperation.__init__(self, probability)
        self.max_shear_left = max_shear_left
        self.max_shear_right = max_shear_right

    def perform_operation(self, images):

        width, height = images[0].size

        angle_to_shear = int(random.uniform((abs(self.max_shear_left) * -1) - 1, self.max_shear_right + 1))
        if angle_to_shear != -1: angle_to_shear += 1

        directions = ["x", "y"]
        direction = random.choice(directions)

        def do(image):

            phi = math.tan(math.radians(angle_to_shear))

            if direction == "x":

                shift_in_pixels = phi * height

                if shift_in_pixels > 0:
                    shift_in_pixels = math.ceil(shift_in_pixels)
                else:
                    shift_in_pixels = math.floor(shift_in_pixels)

                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1

                # Note: PIL expects the inverse scale, so 1/scale_factor for example.
                transform_matrix = (1, phi, -matrix_offset,
                                    0, 1, 0)

                image = image.transform((int(round(width + shift_in_pixels)), height),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((abs(shift_in_pixels), 0, width, height))

                return image.resize((width, height), resample=Image.BICUBIC)

            elif direction == "y":
                shift_in_pixels = phi * width

                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1

                transform_matrix = (1, 0, 0,
                                    phi, 1, -matrix_offset)

                image = image.transform((width, int(round(height + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((0, abs(shift_in_pixels), width, height))

                return image.resize((width, height), resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Distort(OverallOperation):
    """
    This class performs randomised, elastic distortions on images.
    """

    def __init__(self, probability, grid_width, grid_height, magnitude):

        OverallOperation.__init__(self, probability)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        # TODO: Implement non-random magnitude.
        self.randomise_magnitude = True

    def perform_operation(self, images):

        w, h = images[0].size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        if horizontal_tiles > w:
            horizontal_tiles = w
        if vertical_tiles > h:
            vertical_tiles = h

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = random.randint(-self.magnitude, self.magnitude)
            dy = random.randint(-self.magnitude, self.magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        def do(image):

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class GaussianDistortion(OverallOperation):
    """
    This class performs randomised, elastic gaussian distortions on images.
    """

    def __init__(self, probability, grid_width, grid_height, magnitude, corner, method, mex, mey, sdx, sdy):

        OverallOperation.__init__(self, probability)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        self.randomise_magnitude = True
        self.corner = corner
        self.method = method
        self.mex = mex
        self.mey = mey
        self.sdx = sdx
        self.sdy = sdy

    def perform_operation(self, images):

        w, h = images[0].size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        if horizontal_tiles > w:
            horizontal_tiles = w
        if vertical_tiles > h:
            vertical_tiles = h

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        def sigmoidf(x, y, sdx=0.05, sdy=0.05, mex=0.5, mey=0.5, const=1):
            sigmoid = lambda x1, y1: (
                        const * (math.exp(-(((x1 - mex) ** 2) / sdx + ((y1 - mey) ** 2) / sdy))) + max(0, -const) - max(
                    0, const))
            xl = np.linspace(0, 1)
            yl = np.linspace(0, 1)
            X, Y = np.meshgrid(xl, yl)

            Z = np.vectorize(sigmoid)(X, Y)
            mino = np.amin(Z)
            maxo = np.amax(Z)
            res = sigmoid(x, y)
            res = max(((((res - mino) * (1 - 0)) / (maxo - mino)) + 0), 0.01) * self.magnitude
            return res

        def corner(x, y, corner="ul", method="out", sdx=0.05, sdy=0.05, mex=0.5, mey=0.5):
            ll = {'dr': (0, 0.5, 0, 0.5), 'dl': (0.5, 1, 0, 0.5), 'ur': (0, 0.5, 0.5, 1), 'ul': (0.5, 1, 0.5, 1),
                  'bell': (0, 1, 0, 1)}
            new_c = ll[corner]
            new_x = (((x - 0) * (new_c[1] - new_c[0])) / (1 - 0)) + new_c[0]
            new_y = (((y - 0) * (new_c[3] - new_c[2])) / (1 - 0)) + new_c[2]
            if method == "in":
                const = 1
            else:
                if method == "out":
                    const = - 1
                else:
                    const = 1
            res = sigmoidf(x=new_x, y=new_y, sdx=sdx, sdy=sdy, mex=mex, mey=mey, const=const)

            return res

        def do(image):

            for a, b, c, d in polygon_indices:
                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]

                sigmax = corner(x=x3 / w, y=y3 / h, corner=self.corner, method=self.method, sdx=self.sdx, sdy=self.sdy,
                                mex=self.mex, mey=self.mey)
                dx = np.random.normal(0, sigmax, 1)[0]
                dy = np.random.normal(0, sigmax, 1)[0]
                polygons[a] = [x1, y1,
                               x2, y2,
                               x3 + dx, y3 + dy,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
                polygons[b] = [x1, y1,
                               x2 + dx, y2 + dy,
                               x3, y3,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
                polygons[c] = [x1, y1,
                               x2, y2,
                               x3, y3,
                               x4 + dx, y4 + dy]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
                polygons[d] = [x1 + dx, y1 + dy,
                               x2, y2,
                               x3, y3,
                               x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class OpticalDistortion(OverallOperation):
    """
    光学失真
    """
    def __init__(self, probability, distort_limit, shift_limit):
        OverallOperation.__init__(self, probability)
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit

    def perform_operation(self, images):
        k = random.uniform(-self.distort_limit, self.distort_limit)
        dx = round(random.uniform(-self.shift_limit,self.shift_limit))
        dy = round(random.uniform(-self.shift_limit,self.shift_limit))
        interpolation = cv2.INTER_LINEAR
        border_mode = cv2.BORDER_REFLECT_101
        value = None

        def do(image):
            image = np.array(image)
            height, width = image.shape[:2]

            fx = width
            fy = width

            cx = width * 0.5 + dx
            cy = height * 0.5 + dy

            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
            map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height),
                                                     cv2.CV_32FC1)
            image = cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=value)
            image = Image.fromarray(image)
            return image

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images

class Zoom(OverallOperation):
    """
    This class is used to enlarge images (to zoom) but to return a cropped
    region of the zoomed image of the same size as the original image.
    """

    def __init__(self, probability, min_factor, max_factor):
        OverallOperation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        factor = round(random.uniform(self.min_factor, self.max_factor), 2)

        def do(image):
            w, h = image.size

            image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                         int(round(image.size[1] * factor))),
                                        resample=Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            return image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                      floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) + (float(h) / 2))))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class RandomErasing(OverallOperation):
    """
    Class that performs Random Erasing, an augmentation technique described
    in `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
    by Zhong et al. To quote the authors, random erasing:
    "*... randomly selects a rectangle region in an image, and erases its
    pixels with random values.*"
    Exactly this is provided by this class.
    Random Erasing can make a trained neural network more robust to occlusion.
    """

    def __init__(self, probability, rectangle_area):

        OverallOperation.__init__(self, probability)
        self.rectangle_area = rectangle_area

    def perform_operation(self, images):

        def do(image):

            w, h = image.size

            w_occlusion_max = int(w * self.rectangle_area)
            h_occlusion_max = int(h * self.rectangle_area)

            w_occlusion_min = int(w * 0.1)
            h_occlusion_min = int(h * 0.1)

            w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
            h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

            if len(image.getbands()) == 1:
                rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
            else:
                rectangle = Image.fromarray(
                    np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 255))

            random_position_x = random.randint(0, w - w_occlusion)
            random_position_y = random.randint(0, h - h_occlusion)

            image.paste(rectangle, (random_position_x, random_position_y))

            return image

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Custom(OverallOperation):
    """
    Class that allows for a custom operation to be performed using Augmentor's
    standard :class:`~Augmentor.Pipeline.Pipeline` object.
    """

    def __init__(self, probability, custom_function, **function_arguments):
        """
        Creates a custom operation that can be added to a pipeline.
        To add a custom operation you can instantiate this class, passing
        a function pointer, :attr:`custom_function`, followed by an
        arbitrarily long list keyword arguments, :attr:`\*\*function_arguments`.
        .. seealso:: The :func:`~Augmentor.Pipeline.Pipeline.add_operation`
         function.
        :param probability: The probability that the operation will be
         performed.
        :param custom_function: The name of the function that performs your
         custom code. Must return an Image object and accept an Image object
         as its first parameter.
        :param function_arguments: The arguments for your custom operation's
         code.
        """
        OverallOperation.__init__(self, probability)
        self.custom_function = custom_function
        self.function_arguments = function_arguments

    def __str__(self):
        return "Custom (" + self.custom_function.__name__ + ")"

    def perform_operation(self, images):
        return self.function_name(images, **self.function_arguments)


class Mixup(OverallOperation):
    """
    Implements the *mixup* augmentation method, as described in:
    Zhang et al. (2018), *mixup*: Beyond Empirical Risk Minimization,
    arXiv:1710.09412
    See `http://arxiv.org/abs/1710.09412 <http://arxiv.org/abs/1710.09412>`_
    for details.
    Also see `https://github.com/facebookresearch/mixup-cifar10 <https://github.com/facebookresearch/mixup-cifar10>`_
    for code which was followed to create this functionality in Augmentor.
    """

    def __init__(self, probability, alpha=0.4):
        """
        .. note:: Not yet enabled!
            This function is currently implemented but not **enabled**, as it
            requires each image's label in order to operate - something which
            Augmentor was not designed to handle.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param alpha: The alpha parameter controls the strength of the
         interpolation between image-label pairs. It's value can be any value
         greater than 0. A smaller value for :attr:`alpha` results in more
         values closer to 0 or 1, meaning the *mixup* is more often closer to
         either of the images in the pair. Its value is set to 0.4 by default.
        :type probability: Float
        :type alpha: Float
        """
        OverallOperation.__init__(self, probability)
        self.alpha = alpha

    def perform_operation(self, images):

        if self.alpha > 0:
            lambda_value = np.random.beta(self.alpha, self.alpha)
        else:
            lambda_value = 1

        def do(image1, image2, y1, y2):

            image1 = np.asarray(image1)
            image1 = image1.astype('float32')

            image2 = np.asarray(image2)
            image2 = image2.astype('float32')

            mixup_x = lambda_value * image1 + (1 - lambda_value) * image2

            mixup_y = lambda_value * y1 + (1 - lambda_value) * y2

            return mixup_x, mixup_y

        augmented_images = []

        y1 = np.array([0.0, 1.0])
        y2 = np.array([1.0, 0.0])

        for image in images:
            augmented_images.append(do(image, image, y1, y2))

        return augmented_images
