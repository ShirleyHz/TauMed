import TauMed.OverallOperations
import TauMed.LesionRecognition
# import OverallOperations
# import LesionRecognition

from PIL import Image, ImageEnhance
import numpy as np


class LesionOperations(object):
    """
    对于医疗图像的病灶区域扩增处理操作
    """

    def __init__(self, probability):
        self.probability = probability

    def __str__(self):
        return self.__class__.__name__

    def perform_operation(self, images):
        raise RuntimeError("Illegal call to base class.")


class LesionErasing(LesionOperations):
    """
    病灶区域擦除
    """

    def perform_operation(self, images):
        def do(image):
            start_x, start_y, end_x, end_y = TauMed.getLesionPos(image)
            rectangle = Image.fromarray(np.uint8(np.random.rand(int(end_x - start_x), int(end_y - start_y)) * 255))

            image.paste(rectangle, (int(start_x), int(start_y)))
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class LesionRandomBrightness(LesionOperations):
    """
    病灶区域随机改变亮度
    """

    def __init__(self, probability, min_factor, max_factor):
        LesionOperations.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def perform_operation(self, images):
        factor = np.random.uniform(self.min_factor, self.max_factor)

        def do(image):
            start_x, start_y, end_x, end_y = TauMed.getLesionPos(image)

            box = (start_x, start_y, end_x, end_y)
            paste_image = image.crop(box)
            paste_image = ImageEnhance.Brightness(paste_image).enhance(factor)

            image.paste(paste_image, (int(start_x), int(start_y)))
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class LesionPaste(LesionOperations):
    """
    在原始图像中添加多个病灶
    lesion_num: 病灶数
    """
    def __init__(self, probability, lesion_num):
        LesionOperations.__init__(self, probability)
        self.lesion_num = lesion_num

    def perform_operation(self, images):

        def do(image):
            start_x, start_y, end_x, end_y = TauMed.getLesionPos(image)
            box = (start_x, start_y, end_x, end_y)
            lesion_image = image.crop(box)
            w, h = image.size
            count = self.lesion_num
            while count > 0:
                x = np.random.randint(w/4, 3*w/4)
                y = np.random.randint(h/4, 3*h/4)
                image.paste(lesion_image, (x, y))
                count -= 1
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class LesionExpansion(LesionOperations):
    """
    扩大病灶区域，即增大病变范围
    """
    def __init__(self, probability, magnitude):
        LesionOperations.__init__(self, probability)
        self.magnitude = magnitude

    def perform_operation(self, images):

        def do(image):
            start_x, start_y, end_x, end_y = TauMed.getLesionPos(image)
            box = (start_x, start_y, end_x, end_y)
            lesion_image = image.crop(box)
            w, h = lesion_image.size
            new_w = int(w*self.magnitude)
            new_h = int(h*self.magnitude)
            if new_w > 0 and new_h > 0:
                lesion_image = lesion_image.resize((int(w*self.magnitude), int(h*self.magnitude)), Image.ANTIALIAS)
                image.paste(lesion_image, (int(start_x), int(start_y)))
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class AddNonLesionNoises(LesionOperations):
    """
    在非病灶区域添加椒盐噪声
    """
    def __init__(self, probability):
        LesionOperations.__init__(self, probability)

    def perform_operation(self, images):

        def do(image):

            start_x, start_y, end_x, end_y = TauMed.getLesionPos(image)
            box = (start_x, start_y, end_x, end_y)
            lesion_image = image.crop(box)
            image = np.array(image)
            rows, cols, dims = image.shape

            # 随机生成5000个椒盐噪声
            for i in range(10000):
                x = np.random.randint(0, rows)
                y = np.random.randint(0, cols)
                image[x, y, :] = 255

            image = Image.fromarray(image)
            image.paste(lesion_image, (int(start_x), int(start_y)))
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class NonLesionContrast(LesionOperations):
    """
    非病灶区域提高对比度
    """

    def __init__(self, probability, factor):
        LesionOperations.__init__(self, probability)
        self.factor = factor

    def perform_operation(self, images):

        def do(image):
            start_x, start_y, end_x, end_y = TauMed.getLesionPos(image)
            box = (start_x, start_y, end_x, end_y)
            lesion_image = image.crop(box)

            image = ImageEnhance.Contrast(image).enhance(self.factor)
            image.paste(lesion_image, (int(start_x), int(start_y)))
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class NonLesionSharpness(LesionOperations):
    """
    非病灶区域改变锐度
    """

    def __init__(self, probability, factor):
        LesionOperations.__init__(self, probability)
        self.factor = factor

    def perform_operation(self, images):

        def do(image):
            start_x, start_y, end_x, end_y = TauMed.getLesionPos(image)
            box = (start_x, start_y, end_x, end_y)
            lesion_image = image.crop(box)

            image = ImageEnhance.Sharpness(image).enhance(self.factor)
            image.paste(lesion_image, (int(start_x), int(start_y)))
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


