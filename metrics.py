import math
import torch
from torcheval.metrics.functional import binary_auprc

def _binarize_act(neuron_act, alpha):
    """
    added alpha=False option to not binarize at all
    """
    if alpha==False:
        return neuron_act
    else:
        with torch.no_grad():
            k = math.ceil(alpha*len(neuron_act))
            vals, ids = torch.topk(neuron_act, k=k, dim=0)
            cutoff = vals[-1]
            onehot_neuron_act = (neuron_act >= cutoff).float()
    return onehot_neuron_act

def _normalize(tensor):
    """
    tensor: n x d
    normalizes each n dimensional vector to have mean 0 and standard deviation 1
    """
    norm_tensor = tensor - torch.mean(tensor, dim=0, keepdims=True)
    norm_tensor = norm_tensor/torch.clamp(torch.std(norm_tensor, dim=0, keepdims=True), min=1e-9)
    return norm_tensor

def _get_ranks(tensor, noise_mag=1e-7):
    """
    tensor: n x d
    Returns the ranks of elements in a tensor along the 0th dimenstion, with 0 for the smallest element
    adds small random noise to avoid ties, corresponds to randomly selecting the order for tied elements
    """
    noise = noise_mag*torch.rand(tensor.shape, device=tensor.device)
    ranks = torch.argsort(tensor+noise, dim=0, descending=False)
    ranks = torch.argsort(ranks, dim=0) #smallest values have smallest ranks
    return ranks

def recall(neuron_act, concept_prob, alpha):
    """
    same as topk_measure before
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: top fraction of inputs looked at
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        onehot_neuron_act = _binarize_act(neuron_act, alpha)
        onehot_concept_prob = (concept_prob >= 0.5).float()
        
        tp = onehot_neuron_act.T @ onehot_concept_prob
        sum_act = torch.sum(onehot_neuron_act, dim=0).unsqueeze(1)
        similarities = tp/sum_act
    return similarities

def precision(neuron_act, concept_prob, alpha):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: float, top fraction of inputs considered active
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        onehot_neuron_act = _binarize_act(neuron_act, alpha)
        onehot_concept_prob = (concept_prob >= 0.5).float()
        
        tp = onehot_neuron_act.T @ onehot_concept_prob
        sum_concept_prob = torch.sum(onehot_concept_prob, dim=0, keepdims=True)
        similarity = tp/torch.clamp(sum_concept_prob, min=1)
        return similarity

def f1_score(neuron_act, concept_prob, alpha):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: float, top fraction of inputs considered active
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        onehot_neuron_act = _binarize_act(neuron_act, alpha)
        onehot_concept_prob = (concept_prob >= 0.5).float()
        
        tp = onehot_neuron_act.T @ onehot_concept_prob
        fp = onehot_neuron_act.T @ (1-onehot_concept_prob)
        fn = (1-onehot_neuron_act.T) @ onehot_concept_prob
        f1 = 2*tp/(2*tp+fp+fn)
        return f1

def iou(neuron_act, concept_prob, alpha=0.01):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: float, top fraction of inputs looked at
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        onehot_neuron_act = _binarize_act(neuron_act, alpha)
        onehot_concept_prob = (concept_prob >= 0.5).float()

        intersections = onehot_neuron_act.T @ onehot_concept_prob
        unions = (torch.sum(onehot_neuron_act, dim=0, keepdims=True)).T + torch.sum(onehot_concept_prob, dim=0, keepdims=True) - intersections 
        similarities = intersections/unions
    return similarities

def accuracy(neuron_act, concept_prob, alpha):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: float, top fraction of inputs considered active
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    #binarize concept data
    with torch.no_grad():
        onehot_concept_prob = (concept_prob >= 0.5).float()
        onehot_neuron_act = _binarize_act(neuron_act, alpha)

        intersections = onehot_neuron_act.T @ onehot_concept_prob
        neither = (1-onehot_neuron_act).T @ (1-onehot_concept_prob)
        accs = (intersections+neither)/len(neuron_act)
        return accs

def balanced_accuracy(neuron_act, concept_prob, alpha):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: float, top fraction of inputs considered active
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    #binarize concept data
    with torch.no_grad():
        onehot_concept_prob = (concept_prob >= 0.5).float()
        onehot_neuron_act = _binarize_act(neuron_act, alpha)

        intersections = (onehot_neuron_act.T @ onehot_concept_prob)/(torch.clamp(torch.sum(onehot_neuron_act, dim=0, keepdims=True).T, min=1))
        neither = ((1-onehot_neuron_act).T @ (1-onehot_concept_prob))/(torch.clamp(torch.sum(1-onehot_neuron_act, dim=0, keepdims=True).T, min=1))
        accs = (intersections+neither)/2
        return accs

def inverse_balanced_accuracy(neuron_act, concept_prob, alpha):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: float, top fraction of inputs considered active
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    #binarize concept data
    with torch.no_grad():
        onehot_concept_prob = (concept_prob >= 0.5).float()
        onehot_neuron_act = _binarize_act(neuron_act, alpha)

        intersections = (onehot_neuron_act.T @ onehot_concept_prob)/(torch.clamp(torch.sum(onehot_concept_prob, dim=0, keepdims=True), min=1))
        neither = ((1-onehot_neuron_act).T @ (1-onehot_concept_prob))/(torch.clamp(torch.sum(1-onehot_concept_prob, dim=0, keepdims=True), min=1))
        accs = (intersections+neither)/2
        return accs

def auc(neuron_act, concept_prob, alpha):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    alpha: float, top fraction of inputs considered active
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    much more efficient way to calculate the set based on ranks of neuron_act. 
    On 50000 x 1000 inputs takes 6s to calculate vs 20mins for original.
    """
    #binarize concept data
    onehot_neuron_act = _binarize_act(neuron_act, alpha)
    ranks = _get_ranks(concept_prob)

    with torch.no_grad():
        similarities = torch.zeros([neuron_act.shape[1], concept_prob.shape[1]]).to(neuron_act.device)
        for i in range(concept_prob.shape[1]):
            curr_ranks = ranks[:, i:i+1]*(onehot_neuron_act==1)
            n_active = torch.sum(onehot_neuron_act==1, dim=0)
            n_inactive = torch.sum(onehot_neuron_act==0, dim=0)
            auc = (torch.sum(curr_ranks, dim=0) - n_active*(n_active-1)/2)/(torch.clip(n_active*n_inactive, min=1))
            similarities[:, i] =  auc
    return similarities

def inverse_auc(neuron_act, concept_prob):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    much more efficient way to calculate the set based on ranks of neuron_act. 
    On 50000 x 1000 inputs takes 6s to calculate vs 20mins for original.
    """
    #binarize concept data
    onehot_concept_prob = (concept_prob >= 0.5).int()
    ranks = _get_ranks(neuron_act)
    with torch.no_grad():
        similarities = torch.zeros([neuron_act.shape[1], onehot_concept_prob.shape[1]]).to(neuron_act.device)
        for i in range(neuron_act.shape[1]):
            curr_ranks = ranks[:, i:i+1]*(onehot_concept_prob==1)
            n_concept = torch.sum(onehot_concept_prob==1, dim=0)
            n_no_concept = torch.sum(onehot_concept_prob==0, dim=0)
            auc = (torch.sum(curr_ranks, dim=0) - n_concept*(n_concept-1)/2)/(torch.clip(n_concept*n_no_concept, min=1))
            similarities[i] =  auc
    return similarities

def correlation(neuron_act, concept_prob):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        norm_act = _normalize(neuron_act)
        norm_concept = _normalize(concept_prob)
        #Need to divide since matrix product calculates sum, we want mean 
        similarities = (norm_act.T@norm_concept)/len(neuron_act)
        return similarities

def correlation_top_and_random(neuron_act, concept_prob, k=25, alpha=0.002):
    """
    Calculates correlation on a mix of k randomly selected inputs and k random inputs from the top alpha activations
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    similarities = []
    for i in range(neuron_act.shape[1]):
        n_top = math.ceil(alpha*len(neuron_act))
        curr_act = neuron_act[:, i].clone() #|D|
        top_vals, top_ids = torch.topk(curr_act, dim=0, k=n_top)
        top_ids = top_ids[torch.randperm(n_top)[:k]]
        rand_ids = torch.randperm(len(neuron_act), device=neuron_act.device)[:k]
        all_ids = torch.cat([top_ids, rand_ids], dim=0)
        sim = correlation(curr_act[all_ids].unsqueeze(1), concept_prob[all_ids])
        similarities.append(sim)
    similarities = torch.cat(similarities, dim=0)
    return similarities

def correlation_top_and_random_binary(neuron_act, concept_prob, k=25):
    """
    Calculates correlation on a mix of k randomly selected inputs and k random inputs from the top alpha activations
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    similarities = []
    for i in range(neuron_act.shape[1]):
        curr_act = neuron_act[:, i].clone() #|D|
        top_ids = torch.nonzero(curr_act == 1, as_tuple=True)
        top_ids = top_ids[torch.randperm(len(top_ids))[:k]]
        rand_ids = torch.randperm(len(neuron_act), device=neuron_act.device)[:k]
        all_ids = torch.cat([top_ids, rand_ids], dim=0)
        sim = correlation(curr_act[all_ids].unsqueeze(1), concept_prob[all_ids])
        similarities.append(sim)
    similarities = torch.cat(similarities, dim=0)
    return similarities

def spearman_correlation(neuron_act, concept_prob):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    #add small noise to deal with ties correctly
    neuron_ranks = _get_ranks(neuron_act)
    concept_ranks = _get_ranks(concept_prob)
    
    return correlation(neuron_ranks.float(), concept_ranks.float())

def spearman_correlation_top_and_random(neuron_act, concept_prob, k=25, alpha=0.002):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    similarities = []
    for i in range(neuron_act.shape[1]):
        n_top = math.ceil(alpha*len(neuron_act))
        curr_act = neuron_act[:, i].clone() #|D|
        top_vals, top_ids = torch.topk(curr_act, dim=0, k=n_top)
        top_ids = top_ids[torch.randperm(n_top)[:k]]
        rand_ids = torch.randperm(len(neuron_act), device=neuron_act.device)[:k]
        all_ids = torch.cat([top_ids, rand_ids], dim=0)
        sim = spearman_correlation(curr_act[all_ids].unsqueeze(1), concept_prob[all_ids])
        similarities.append(sim)
    similarities = torch.cat(similarities, dim=0)
    return similarities

def spearman_correlation_top_and_random_binary(neuron_act, concept_prob, k=25):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    similarities = []
    for i in range(neuron_act.shape[1]):
        curr_act = neuron_act[:, i].clone() #|D|
        top_ids = torch.nonzero(curr_act == 1, as_tuple=True)
        top_ids = top_ids[torch.randperm(len(top_ids))[:k]]
        rand_ids = torch.randperm(len(neuron_act), device=neuron_act.device)[:k]
        all_ids = torch.cat([top_ids, rand_ids], dim=0)
        sim = spearman_correlation(curr_act[all_ids].unsqueeze(1), concept_prob[all_ids])
        similarities.append(sim)
    similarities = torch.cat(similarities, dim=0)
    return similarities

def cos_sim(neuron_act, concept_prob):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        norm_act = neuron_act/torch.clamp(torch.norm(neuron_act, dim=0, p=2, keepdim=True), min=1e-9)
        norm_concept = concept_prob/torch.clamp(torch.norm(concept_prob, dim=0, p=2, keepdim=True), min=1e-9)
        return norm_act.T @ norm_concept

def wpmi(neuron_act, concept_prob, alpha, lam=1):
    """
    Pointwise mutual information
    same as simulation cross_entropy if lambda=0
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    lam: 
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        onehot_neuron_act = _binarize_act(neuron_act, alpha)
        log_likelihoods = (onehot_neuron_act.T)@torch.log(torch.clamp(concept_prob, min=1e-8))
        concept_likelihood = torch.log(torch.mean(concept_prob, dim=0, keepdims=True))
        n_neuron_act = torch.sum(onehot_neuron_act, dim=0).unsqueeze(1)
        log_likelihoods = log_likelihoods/torch.clamp(n_neuron_act, min=1)
        similarities = log_likelihoods - lam*concept_likelihood
        return similarities

def mad(neuron_act, concept_prob):
    """
    Mean activation difference
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    with torch.no_grad():
        onehot_concept_prob = (concept_prob >= 0.5).int()
        n_concept = torch.sum(onehot_concept_prob, dim=0)
        n_no_concept = torch.sum(1-onehot_concept_prob, dim=0)
        similarities = torch.zeros([neuron_act.shape[1], concept_prob.shape[1]]).to(neuron_act.device) #n_neurons x n_concepts
        for i in range(concept_prob.shape[1]):
            #should use onehot_concept_prob instead
            conc_mean = torch.sum(neuron_act * onehot_concept_prob[:, i:i+1], dim=0)/torch.clamp(n_concept[i], min=1) #n_neurons 
            no_conc_mean = torch.sum(neuron_act * (1-onehot_concept_prob[:, i:i+1]), dim=0)/torch.clamp(n_no_concept[i], min=1)
            #print(conc_mean.shape, no_conc_mean.shape, similarities.shape)
            similarities[:, i] = conc_mean - no_conc_mean
        return similarities

def auprc(neuron_act, concept_prob, alpha):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    #binarize concept data
    onehot_neuron_act = _binarize_act(neuron_act, alpha)
    #onehot_concept_prob = (concept_prob >= 0.5).int()
    similarities = torch.zeros([neuron_act.shape[1], concept_prob.shape[1]], dtype=float, device=neuron_act.device)
    #noise doesn't make a difference
    #noisy_act = neuron_act + torch.rand(neuron_act.shape, device=neuron_act.device)
    with torch.no_grad():
        for i in range(neuron_act.shape[1]):
            for j in range(concept_prob.shape[1]):
                similarities[i, j] = binary_auprc(concept_prob[:, j], onehot_neuron_act[:, i])
            #similarities[i] = binary_auprc(neuron_act[:, i:i+1].expand(-1, concept_prob.shape[1]).mT, onehot_concept_prob.mT, num_tasks=concept_prob.shape[1])
    return similarities

def inverse_auprc(neuron_act, concept_prob):
    """
    neuron_act: |D| x n_neurons
    concept_prob: |D| x n_concepts
    returns (n_neurons x n_concepts) similarity vector of how similar the neuron is to each concept
    """
    #binarize concept data
    onehot_concept_prob = (concept_prob >= 0.5).int()
    similarities = torch.zeros([neuron_act.shape[1], concept_prob.shape[1]], dtype=float, device=neuron_act.device)
    #noise doesn't make a difference
    #noisy_act = neuron_act + torch.rand(neuron_act.shape, device=neuron_act.device)
    with torch.no_grad():
        for i in range(neuron_act.shape[1]):
            for j in range(concept_prob.shape[1]):
                similarities[i, j] = binary_auprc(neuron_act[:, i], onehot_concept_prob[:, j])
            #similarities[i] = binary_auprc(neuron_act[:, i:i+1].expand(-1, concept_prob.shape[1]).mT, onehot_concept_prob.mT, num_tasks=concept_prob.shape[1])
    return similarities


### Combination Metrics for Appendix

def _harmonic_mean(tensor1, tensor2):
    return 2*tensor1*tensor2/(tensor1 + tensor2)

def combined_auc(neuron_act, concept_prob, alpha):
    return _harmonic_mean(auc(neuron_act, concept_prob, alpha), inverse_auc(neuron_act, concept_prob))

def combined_balanced_acc(neuron_act, concept_prob, alpha):
    return _harmonic_mean(balanced_accuracy(neuron_act, concept_prob, alpha), inverse_balanced_accuracy(neuron_act, concept_prob, alpha))

def recall_auc(neuron_act, concept_prob, alpha):
    return _harmonic_mean(recall(neuron_act, concept_prob, alpha), auc(neuron_act, concept_prob, alpha))

def recall_inv_auc(neuron_act, concept_prob, alpha):
    return _harmonic_mean(recall(neuron_act, concept_prob, alpha), inverse_auc(neuron_act, concept_prob))

def precision_bal_acc(neuron_act, concept_prob, alpha):
    return _harmonic_mean(precision(neuron_act, concept_prob, alpha), balanced_accuracy(neuron_act, concept_prob, alpha))

def precision_inverse_bal_acc(neuron_act, concept_prob, alpha):
    return _harmonic_mean(precision(neuron_act, concept_prob, alpha), inverse_balanced_accuracy(neuron_act, concept_prob, alpha))