import torch
import torch.nn.functional as F
import logging
import numpy as np
import sklearn.metrics.pairwise as smp

logger = logging.getLogger('logger')
    
def foolsgold(helper, local_updates, target_model, history, train_data_sets, epsilon=1e-5):
    
    ## update history information
    his_matrix = []
    for i in range(helper.params['no_models']):
        client_id = train_data_sets[i]
        his_i = (torch.cat([p.view(-1) for name, p in local_updates[i].items()])).cpu().numpy().flatten() 
        his_matrix = np.append(his_matrix, his_i)
        
        if client_id not in history:
            history[client_id] = his_i
        else:
            history[client_id] = history[client_id] + his_i
    
    
    his_matrix = np.reshape(his_matrix, (helper.params['no_models'], -1))
    logger.info("FoolsGold: Finish loading history updates")
    
    cs = smp.cosine_similarity(his_matrix) - np.eye(helper.params['no_models'])
    maxcs = np.max(cs, axis=1)
    for i in range(helper.params['no_models']):
        for j in range(helper.params['no_models']):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    
    logger.info("FoolsGold: Calculate max similarities")
    
    # Pardoning
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    wv = F.softmax(torch.tensor(wv), dim=0)
    
    # Federated SGD iteration
    logger.info(f"FoolsGold: Accumulation with lr {wv.numpy()}")
    weight_accumulator = dict()
    for name, data in local_updates[0].items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)
    
    for i in range(helper.params['no_models']):
        for name, data in local_updates[i].items():
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name].add_((wv[i]*data).to(weight_accumulator[name].dtype))

    non_zero = np.count_nonzero(wv)

    for name, data in target_model.state_dict().items():
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue

        update_per_layer = weight_accumulator[name] * helper.params["eta"]
        data.add_(update_per_layer.to(data.dtype))