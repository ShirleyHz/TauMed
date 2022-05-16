import numpy as np
from Models.panet import PanNet
import torch
import warnings

warnings.filterwarnings('ignore')


def getLesionPos(image):
    """
    通过训练好的PaNet模型获取病灶区域
    :param image: 输入图像
    :return: 病灶区域坐标
    """
    # if image.format == 'GIF' or image.format == 'gif':
    #     raise ValueError("GIF Error")

    model = PanNet()
    # model.load_state_dict(torch.load('/TauMed/TauMed/Models/panet_model_4.dms')) Docker
    model.load_state_dict(torch.load('/Users/vivi/PycharmProjects/TauMed/Models/panet_model_4.dms')) # 本地
    # image = image.astype(np.float32)
    image = image.resize((512, 512))
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    image = torch.unsqueeze(image, dim=0).float()
    bbox_pred = model(image)
    bbox_pred = bbox_pred.tolist()
    start_x = bbox_pred[0][0][0]
    start_y = bbox_pred[0][0][1]
    end_x = bbox_pred[0][0][2]
    end_y = bbox_pred[0][0][3]
    return start_x, start_y, end_x, end_y
