# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 21:07
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchvision.models as models


def contentFunc():
    conv_3_3_layer = 14
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.cuda()
    model = nn.Sequential()
    model = model.cuda()
    for i, layer in enumerate(list(cnn)):
        model.add_module(str(i), layer)
        if i == conv_3_3_layer:
            break
    return model


class PerceptualLoss:

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


def perceptual_loss(input: torch.Tensor, target: torch.Tensor):
    P = PerceptualLoss(nn.MSELoss())
    input_ = input.repeat(1, 3, 1, 1)
    target_ = target.repeat(1, 3, 1, 1)
    return 100 * P.get_loss(input_, target_)


def correlation(input, target):
    input_vector =  input.reshape((1, -1))
    target_vector = target.reshape((1, -1))
    return torch.corrcoef(torch.cat([input_vector, target_vector], dim=0))[0, 1]
