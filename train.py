import logging
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.add_trigger import add_pixel_pattern, add_pixel_pattern_dba
from utils.utils import PerturbedGradientDescent, Attention
from collections import OrderedDict

logger = logging.getLogger("logger")
criterion = torch.nn.CrossEntropyLoss()

def sars_ptrain(helper, epoch, train_data_sets, target_model):
    target_model_copy = copy.deepcopy(target_model)
    for client_id in train_data_sets:

        local_model = helper.local_models[client_id]
        local_model = local_model.cuda()
        local_model.train()
        train_data = helper.train_data[client_id]

        t_optimizer = torch.optim.SGD(target_model_copy.parameters(),
                                      lr=helper.params['ST_lr'],
                                      momentum=helper.params['momentum'],
                                      weight_decay=helper.params['decay'])

        for local_epoch in range(1, helper.params['ST_retrain_times'] + 1):
            data_iterator = train_data
            for batch_id, batch in enumerate(data_iterator):
                t_optimizer.zero_grad()
                data, targets = helper.get_batch(batch, evaluation=False)
                attention_map1, attention_map2, attention_map3, attention_map4, output = target_model_copy(data)

                ## SARS(Ours)
                ce_loss = nn.functional.cross_entropy(output, targets)
                attn_loss = Attention(helper.params['p'])
                attn_loss1 = attn_loss(F.interpolate(attention_map1, size=(attention_map2.size(2), attention_map2.size(3))), attention_map2)
                attn_loss2 = attn_loss(F.interpolate(attention_map2, size=(attention_map3.size(2), attention_map3.size(3))), attention_map3)
                attn_loss3 = attn_loss(F.interpolate(attention_map3, size=(attention_map4.size(2), attention_map4.size(3))), attention_map4)

                loss = ce_loss + helper.params['beta'] * (attn_loss1 + attn_loss2 + attn_loss3)

                loss.backward()
                t_optimizer.step()

        optimizer = PerturbedGradientDescent(local_model.parameters(),
                                             lr=helper.params['lr'],
                                             mu=helper.params['mu'])

        for local_epoch in range(1, helper.params['retrain_no_times'] + 1):
            data_iterator = train_data
            for batch_id, batch in enumerate(data_iterator):
                optimizer.zero_grad()
                data, targets = helper.get_batch(batch, evaluation=False)
                _, _, _, _, output = local_model(data)
                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()
                optimizer.step(target_model_copy.parameters())

        helper.local_models[client_id] = local_model

    return helper.local_models


def train(helper, args, epoch, train_data_sets, target_model, is_poison):

    ### local updates
    local_updates = np.array([])

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

    current_number_of_adversaries = 0
    for client_id in train_data_sets:
        if client_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

    for client_id in train_data_sets:
        model = copy.deepcopy(target_model)
        model.train()

        # helper.params['lr'] * 0.998 ** (epoch - 1)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])

        train_data = helper.train_data[client_id]

        local_update = OrderedDict()

        if is_poison and client_id in helper.params['adversary_list']:
            # logger.info('==> poisoning...')
            poisoned_data = helper.poisoned_data_for_train

            retrain_no_times = helper.params['retrain_poison']

            poison_optimizer = torch.optim.SGD(model.parameters(), 
                                               lr=helper.params['poison_lr'],
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])

            for _ in range(1, retrain_no_times + 1):
                data_iterator = train_data

                for batch_id, batch in enumerate(data_iterator):
                    poison_num = min(helper.params['poisoning_per_batch'], len(batch[1]))
                    for i in range(poison_num):
                        if args.attack == 'pixel':
                            batch[0][i] = add_pixel_pattern(helper, batch[0][i])
                        elif args.attack == 'dba':
                            batch[0][i] = add_pixel_pattern_dba(epoch, helper, batch[0][i])
                        
                        batch[1][i] = helper.params['poison_label_swap']

                    data, targets = helper.get_batch(batch, False)
                    poison_optimizer.zero_grad()

                    _, _, _, _, output = model(data)
                    class_loss = nn.functional.cross_entropy(output, targets)

                    distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                    loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss
                    loss.backward()

                    poison_optimizer.step()

        else:
            for _ in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.
                data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(batch, evaluation=False)
                    _, _, _, _, output = model(data)
                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            local_update[name] = (data - target_model.state_dict()[name]).detach()

        local_updates = np.append(local_updates, local_update)

    return local_updates