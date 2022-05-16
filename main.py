#coding: utf-8
"""
面向数据扩增系统的接口
"""
import TauMed
import json
import argparse
import os
import importlib
import sys

importlib.reload(sys)

flip_direction = ["LEFT_RIGHT", "TOP_BUTTON", "RANDOM"]
skew_type = ["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"]
resample_filter = ["BICUBIC", "BILINEAR", "ANTIALIAS", "NEAREST"]
guassian_corner = [ "ul", "ur", "dl", "dr"]
guassian_method = ["in", "out"]


# 图像扩增接口
def image_aug(dataPath, zFile, outputPath, jsonPath):
    # f = open(jsonPath, encoding='utf-8')
    # content = f.read()
    # if content.startswith(u'\ufeff'):
    #     content = content.encode('utf8')[3:].decode('utf8')
    with open(jsonPath) as f:
        operationJson = json.load(f)
    # output_path = operationJson['output_path']
    p = TauMed.Pipeline(source_directory=dataPath, zip_file=zFile, output_directory=outputPath)
    # print(operationJson)
    operations = operationJson['operations']
    for operation in operations:
        # print(operation)
        if operation['type'] == 'Flip':
            p.flip(operation['param'][0], flip_direction[operation['param'][1]])
        elif operation['type'] == 'Rotate':
            p.rotate(operation['param'][0], operation['param'][1], operation['param'][2])
        elif operation['type'] == 'Skew':
            p.skew(operation['param'][0], skew_type[operation['param'][1]], operation['param'][2])
        elif operation['type'] == 'Distort':
            p.random_distortion(operation['param'][0],
                                operation['param'][1],
                                operation['param'][2],
                                operation['param'][3])
        elif operation['type'] == 'GaussianDistortion':
            p.gaussian_distortion(operation['param'][0],
                                  operation['param'][1],
                                  operation['param'][2],
                                  operation['param'][3],
                                  guassian_corner[operation['param'][4]],
                                  guassian_method[operation['param'][5]])
        elif operation['type'] == 'OpticalDistortion':
            p.optical_distortion(operation['param'][0],
                                 operation['param'][1],
                                 operation['param'][2])
        elif operation['type'] == 'RandomCrop':
            p.crop_random(operation['param'][0], operation['param'][1])
        elif operation['type'] == 'CenterCrop':
            p.crop_random(operation['param'][0], operation['param'][1])
        elif operation['type'] == 'Shear':
            p.shear(operation['param'][0], operation['param'][1], operation['param'][2])
        elif operation['type'] == 'RandomErasing':
            p.random_erasing(operation['param'][0], operation['param'][1])
        elif operation['type'] == 'Zoom':
            p.zoom(operation['param'][0], operation['param'][1], operation['param'][2])
        elif operation['type'] == 'HistogramEqualisation':
            p.histogram_equalisation(operation['param'][0])
        elif operation['type'] == 'RandomBrightness':
            p.random_brightness(operation['param'][0],
                                operation['param'][1],
                                operation['param'][2])
        elif operation['type'] == 'Resize':
            p.resize(operation['param'][0],
                     operation['param'][1],
                     operation['param'][2],
                     resample_filter[operation['param'][3]])
        elif operation['type'] == 'LesionErasing':
            p.lesion_erasing(operation['param'][0])
        elif operation['type'] == 'LesionRandomBrightness':
            p.lesion_random_brightness(operation['param'][0],
                                       operation['param'][1],
                                       operation['param'][2])
        elif operation['type'] == 'LesionPaste':
            p.lesion_paste(operation['param'][0], operation['param'][1])
        elif operation['type'] == 'LesionExpansion':
            p.lesion_expansion(operation['param'][0], operation['param'][1])
        elif operation['type'] == 'AddNonLesionNoises':
            p.add_nonlesion_noises(operation['param'][0])
        elif operation['type'] == 'NonLesionContrast':
            p.nonlesion_contrast(operation['param'][0], operation['param'][1])
        elif operation['type'] == 'NonLesionSharpness':
            p.nonlesion_sharpness(operation['param'][0], operation['param'][1])

    p.sample(operationJson['output_num'])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='图像扩增任务')
    # parser.add_argument('id', type=int, help='当前任务id')
    # parser.add_argument('inputPath', type=str, help='数据输入路径')
    # parser.add_argument('outputPath', type=str, help='数据输出路径')
    # parser.add_argument('resultPath', type=str, help='结果文件输出路径')
    # parser.add_argument('jsonPath', type=str, help='配置文件路径')
    #
    # args = parser.parse_args()
    # id = args.id
    # inputPath = args.inputPath
    # outputPath = args.outputPath
    # resultPath = args.resultPath
    # jsonPath = args.jsonPath
    # # zFile = os.path.basename(inputPath)
    # # dataPath = os.path.dirname(inputPath)

    inputPath="Input"
    outputPath="Output"
    jsonPath="content.json"



    for zFile in os.listdir(inputPath):
        if os.path.isfile(os.path.join(inputPath, zFile)) and zFile != '.DS_Store':
            image_aug(inputPath, zFile, outputPath, jsonPath)
            del_list = os.listdir(inputPath + os.sep + "images")
            for f in del_list:
                file_path = os.path.join(inputPath + os.sep + "images", f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
