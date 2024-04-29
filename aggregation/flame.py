import torch
import numpy as np
import hdbscan
import sklearn.metrics.pairwise as smp

def flame(helper, local_updates, target_model, lamda):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in local_updates[0].items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### model filtering
    local_updates_vectors = np.array([])
    for i in range(helper.params['no_models']):
        local_update_i = torch.cat([p.view(-1) for name, p in local_updates[i].items()]).cpu().numpy()
        local_updates_vectors = np.append(local_updates_vectors, local_update_i)

    cosine_distance = smp.cosine_distances(local_updates_vectors.reshape(helper.params['no_models'], -1))
    cluster = hdbscan.HDBSCAN(min_cluster_size=int(helper.params['no_models']//2+1),
                              min_samples=1,
                              allow_single_cluster=True,
                              metric='precomputed').fit(cosine_distance)
    cluster_labels = (cluster.labels_).tolist()

    ### norm clipping
    #### model updates l2-norm
    update_l2_norms = []
    for i in range(len(local_updates)):
        update_l2_norm = helper.model_global_norm(local_updates[i])
        update_l2_norms.append(update_l2_norm)

    st = np.median(update_l2_norms)

    no_models = 0
    print('===> update_l2_norms: ', update_l2_norms)
    print('===> cluster_labels: ', cluster_labels)
    print('===> st: ', st)
    for i in range(len(local_updates)):
        if cluster_labels[i] == -1:
            continue

        no_models += 1
        if st/update_l2_norms[i] < 1:
            for name, data in local_updates[i].items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                    continue
                # print(data.dtype)
                data.mul_(torch.tensor(st / update_l2_norms[i]).to(data.dtype))
                weight_accumulator[name].add_(data)

    helper.average_shrink_models(weight_accumulator, target_model, no_models=no_models)

    ### add noise
    for name, data in target_model.state_dict().items():
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        data.add_(helper.dp_noise(data, sigma=st*lamda).to(data.dtype))