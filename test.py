import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

from torchvision import transforms

from image_helper import ImageHelper
from utils.add_trigger import add_pixel_pattern

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time
import numpy as np

import random

criterion = torch.nn.CrossEntropyLoss()

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

def test(helper, epoch, data_source, model, is_poison=False):
    model.eval()
    model = model.cuda()
    total_loss = 0
    correct = 0
    dataset_size = 0
    
    data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):

        data, targets = helper.get_batch(batch, evaluation=True)
        dataset_size += targets.size(0)
        _, _, _, _, output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = float(correct) / float(dataset_size)
    total_l = total_loss / dataset_size

    # logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
    #             'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch, total_l, correct, dataset_size, acc))

    model.train()
    return total_l, acc


def test_poison(helper, epoch, data_source, model, is_poison=False):
    model.eval()
    model = model.cuda()
    total_loss = 0.0
    correct = 0
    
    data_iterator = data_source
    dataset_size = 0

    for batch_id, batch in enumerate(data_iterator):
        for pos in range(len(batch[1])):
            batch[0][pos] = add_pixel_pattern(helper, batch[0][pos])
            batch[1][pos] = helper.params['poison_label_swap']

        data, targets = helper.get_batch(batch, evaluation=True)

        _, _, _, _, output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        dataset_size += len(batch[1])
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float).data.item()

    acc = float(correct) / float(dataset_size)
    total_l = total_loss / dataset_size
    # logger.info('___Test poisoned: {}, epoch: {}: Average loss: {:.4f}, '
    #             'Accuracy: {}/{} ({:.4f}%)'.format(is_poison, epoch, total_l, correct, dataset_size, acc))

    model.train()
    return total_l, acc