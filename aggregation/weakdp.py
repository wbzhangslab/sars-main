import torch

def weakdp(helper, local_updates, target_model):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in local_updates[0].items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    for i in range(len(local_updates)):
        print(helper.model_global_norm(local_updates[i]))
        for name, data in local_updates[i].items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                continue
            data.mul_(torch.tensor(min(1, helper.params['norm_bound'] / helper.model_global_norm(local_updates[i]))).to(data.dtype))
            weight_accumulator[name].add_((data).to(data.dtype))

    helper.average_shrink_models(weight_accumulator, target_model, no_models=helper.params['no_models'])

    ### add noise
    for name, data in target_model.state_dict().items():
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        data.add_(helper.dp_noise(data, sigma=helper.params['dp_sigmma']).to(data.dtype))