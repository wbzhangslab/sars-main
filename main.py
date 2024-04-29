import os
import argparse
import json
import datetime
import logging
import yaml
import time
import random
import copy

import torch
import numpy as np

from image_helper import ImageHelper
from test import test, test_poison
from train import train, sars_ptrain
from aggregation.fedavg import fedavg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger("logger")
criterion = torch.nn.CrossEntropyLoss()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)

if __name__ == '__main__':
    print('==> Start training...')
    time_start_load_everything = time.time()

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    parser.add_argument('--ptrain_flag', dest='ptrain_flag', type=int, default=1)
    parser.add_argument('--ptrain', dest='ptrain', type=str, default='sars') # ['sars']
    parser.add_argument('--aggregation', dest='aggregation', type=str, default='fedavg')  # ['fedavg'']
    parser.add_argument('--attack', dest='attack', type=str, default='dba') # ['edge-case', 'pixel-patter', 'dba']

    args = parser.parse_args()

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader) # :return dict

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, 
                         params=params_loaded,
                         name=params_loaded.get('type', 'image'))

    helper.load_data()
    helper.create_model()
    global_model = copy.deepcopy(helper.target_model)
    
    lr = helper.params['lr']

    ### Local Clients Setting
    if helper.params['is_poison']:
        helper.params['adversary_list'] = [0] + random.sample(range(helper.params['number_of_total_participants']),
                                                                    helper.params['number_of_adversaries']-1)
        logger.info(f"==> Poisoned following participants: {helper.params['adversary_list']}")

    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    logger.info(f"==> We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.train_data))

    results = {'PM-Main-Acc': list(),
               'GM-Main-Acc': list(),
               'PM-Backdoor-Acc': list(),
               'GM-Backdoor-Acc': list(),
               'PM-Main-Acc-Per-Clients': list(),
               'GM-Main-Acc-Per-Clients': list(),
               'number_of_adversaries': helper.params['number_of_adversaries']}

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    local_updates = None
    dist_list = list()
    history = dict()
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()

        if helper.params['is_poison']:
            ### For poison epoch we put one adversary and other adversaries just stay quiet
            subset_data_chunks = helper.params['adversary_list'] + random.sample(list(set(participant_ids).difference(set(helper.params['adversary_list']))),
                                                                                                                              helper.params['no_models'] - helper.params['number_of_adversaries'])

        else:
            subset_data_chunks = random.sample(participant_ids, helper.params['no_models'])

        logger.info(f'==> Selected models: {subset_data_chunks}')

        ### Personalized
        if args.ptrain == 'sars':
            helper.local_models = sars_ptrain(helper=helper,
                                              epoch=epoch,
                                              train_data_sets=subset_data_chunks,
                                              target_model=helper.target_model)

        ### Local Train
        local_updates = train(helper=helper, 
                              args=args, 
                              epoch=epoch,
                              train_data_sets=subset_data_chunks,
                              target_model=helper.target_model,
                              is_poison=helper.params['is_poison'])

        ### Average the Local model update
        if args.aggregation == 'fedavg':
            fedavg(helper=helper, 
                   local_updates=local_updates,
                   target_model=helper.target_model)

        logger.info(f'==> time spent on training: {time.time() - start_time} sec.')

        ### Evaluating Local Main Task accuracy & Local Backdoor accuracy Each Epoch
        p_main_acc = []
        p_backdoor_acc = []
        g_main_acc = []
        for i in range(helper.params['number_of_total_participants']):
            if i not in helper.params['adversary_list']:

                ## PM C-Acc
                if args.ptrain_flag:
                    _, local_p_main_acc = test(helper=helper,
                                               epoch=epoch,
                                               data_source=helper.eval_data[i],
                                               model=helper.local_models[i],
                                               is_poison=False)
                else:
                    local_p_main_acc = 0.0
                p_main_acc.append(local_p_main_acc)

                ## PM ASR
                if helper.params['is_poison'] and args.ptrain_flag:
                    _, local_p_backdoor_acc = test_poison(helper=helper,
                                                          epoch=epoch,
                                                          data_source=helper.test_data_poison,
                                                          model=helper.local_models[i],
                                                          is_poison=True)

                else:
                    local_p_backdoor_acc = 0.0
                p_backdoor_acc.append(local_p_backdoor_acc)

                
                ## GM C-Acc
                _, local_g_main_acc = test(helper=helper,
                                            epoch=epoch,
                                            data_source=helper.eval_data[i],
                                            model=helper.target_model,
                                            is_poison=False)
                g_main_acc.append(local_g_main_acc)

        ## GM ASR
        if helper.params['is_poison']:
            _, g_backdoor_acc = test_poison(helper=helper,
                                            epoch=epoch,
                                            data_source=helper.test_data_poison,
                                            model=helper.target_model,
                                            is_poison=True)
            results['GM-Backdoor-Acc'].append(g_backdoor_acc)

        else:
            g_backdoor_acc = 0.0
            results['GM-Backdoor-Acc'].append(g_backdoor_acc)

        results['PM-Main-Acc-Per-Clients'].append(p_main_acc)
        results['GM-Main-Acc-Per-Clients'].append(g_main_acc)
        results['PM-Main-Acc'].append(np.array(p_main_acc).mean())
        results['GM-Main-Acc'].append(np.array(g_main_acc).mean())
        results['PM-Backdoor-Acc'].append(np.array(p_backdoor_acc).mean())
        results['GM-Backdoor-Acc'].append(g_backdoor_acc)


        if args.ptrain_flag:
            logger.info('==> Current Epoch {}: '
                        'PM (C-Acc) {:.4f} | '
                        'PM (ASR) {:.4f} | '
                        'GM (ASR) {:.4f} '.format(epoch,
                                                  np.array(p_main_acc).mean(),
                                                  np.array(p_backdoor_acc).mean(),
                                                  g_backdoor_acc))
            
        else:
            logger.info('==> Current Epoch {}: '
                        'GM (C-Acc) {:.4f} | '
                        'GM (ASR) {:.4f} '.format(epoch,
                                                  np.array(g_main_acc).mean(),
                                                  g_backdoor_acc))

    ## Save results
    logger.info('==> Saving all the graphs.')
    logger.info(f"==> This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    if helper.params['results_json']:
        with open('{}/SARS_Dataset({})_Mode(Pixel)_Dir(0.5).json'.format(helper.folder_path, helper.params['dataset']), 'w') as f:
            f.write(json.dumps(results))
    
    attack_flag = None
    algorithm = None
    if helper.params['is_poison'] == 0:
        attack_flag = 'benign'
    
    else:
        attack_flag = args.attack
    
    
    if args.ptrain_flag:
        for i in range(helper.params['number_of_total_participants']):
            if i not in helper.params['adversary_list']:
                torch.save(helper.local_models[i], '{}/{}_LocalMODEL_{}.pth'.format(helper.folder_path, helper.params['dataset'], i))
        algorithm = args.ptrain

    else:
        algorithm = args.aggregation
    
    torch.save(helper.target_model, '{}/{}_{}_{}.pth'.format(helper.folder_path, algorithm, helper.params['dataset'], attack_flag))