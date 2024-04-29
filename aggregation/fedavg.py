import torch

def fedavg(helper, local_updates, target_model):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in local_updates[0].items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    for i in range(len(local_updates)):
        for name, data in local_updates[i].items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name].add_(data)

    helper.average_shrink_models(weight_accumulator, target_model, no_models=helper.params['no_models'])