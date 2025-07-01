import os
import math
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import open_clip

import data_utils

SIGMOID_PARAMS_IMAGENET_SC = {"ViT-SO400M-14-SigLIP-384":{"a":58, "b":-0.13},
                              "ViT-L-16-SigLIP-384":{"a":60, "b":-0.09},
                              "ViT-L-14-336":{"a":58, "b":-0.25},
                              }

#for standard models, using OpenAI weights if available, otherwise OpenCLIP model
CN_TO_CHECKPOINT = {"ViT-SO400M-14-SigLIP-384": "webli",
                    "ViT-L-16-SigLIP-384": "webli",
                    "ViT-g-14": "laion2b_s34b_b88k",
                    "ViT-L-14-336": "openai",
                    "ViT-B-16": "openai",
                    "ViT-L-16-SigLIP-256": "webli",
                    "ViT-B-16-SigLIP-384": "webli",}


def get_target_acts(target_name, dataset_name, target_layer, save_dir, batch_size, device, start_neuron=0, end_neuron=None, pool_mode="avg"):
    model, preprocess = data_utils.get_target_model(target_name, device=device)
    target_data = data_utils.get_data(dataset_name, preprocess)
    
    layer_save_path = '{}/{}_{}/{}/'.format(save_dir, target_name, dataset_name, target_layer)
    activations = save_summary_activations(model, target_data, device, target_layer, batch_size, layer_save_path, pool_mode)
    summary_activations = activations[:, start_neuron:end_neuron]
    return summary_activations


def get_clip_feats(clip_name, dataset_name, concept_set, save_dir,  batch_size, device):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_name.split("_")[-1],
                                                                       pretrained=CN_TO_CHECKPOINT[clip_name],
                                                                       device=device)
    clip_data = data_utils.get_data(dataset_name, clip_preprocess)
    clip_save_name = "{}/{}_{}.pt".format(save_dir, dataset_name, clip_name)
    save_clip_image_features(clip_model, clip_data, clip_save_name, batch_size, device)
    clip_image_features = torch.load(clip_save_name, map_location=device).float()
    
    with open(concept_set, 'r') as f: 
        concept_text = (f.read()).split('\n')
    tokenized_text = open_clip.get_tokenizer(clip_name.split("_")[-1])(concept_text).to(device)
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set.split("/")[-1].split(".")[0], clip_name)
    clip_text_features = save_clip_text_features(clip_model, tokenized_text, text_save_name).float()
    a = SIGMOID_PARAMS_IMAGENET_SC[clip_name]["a"]
    b = SIGMOID_PARAMS_IMAGENET_SC[clip_name]["b"]
    
    with torch.no_grad():
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
        clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
        clip_feats = clip_image_features @ (clip_text_features).T
        clip_feats = torch.nn.functional.sigmoid(a*(clip_feats+b))
    return clip_feats

def get_cub_concept_labels(dataset, device, batch_size):
    concept_set = "data/cub_concept_names.txt"
    with open(concept_set, 'r') as f: 
        concept_text = (f.read()).split('\n')

    cs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False):
            x, y, c = batch
            c = torch.stack(c, dim=1).to(device)
            cs.append(c)
    cs = torch.cat(cs, dim=0).float()
    return cs, concept_text

def get_cub_concept_preds(model, dataset, device, batch_size):
    concept_preds = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False):
            x, y, c = batch
            pred_c = model(x.to(device))
            if type(pred_c) == list:
                pred_c = torch.stack(pred_c, dim=1).squeeze(-1)
            concept_preds.append(pred_c)
    concept_preds = torch.cat(concept_preds, dim=0)
    return concept_preds

def get_llm_ct_ak(model, dataset, device, batch_size, n_neurons=500):
    counts = Counter(dataset.view(-1).cpu().numpy())
    common_toks = [tok for tok, count in counts.most_common(n_neurons)]

    cts = []
    outs = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(dataset)/batch_size))):
            input_tok = dataset[batch_size*i:batch_size*(i+1)]
            
            out = model.forward(input_tok.to(device)) # bs x seq_l x vocab_size
            out = torch.nn.functional.softmax(out, dim=2)

            out_common = out[:, :, common_toks]
            outs.append(out_common[:, :-1, :].cpu()) #don't include last tok of seq since no corresponding gt

            curr_ct = torch.nn.functional.one_hot(input_tok, num_classes=model.tokenizer.vocab_size)
            ct_common = curr_ct[:, :, common_toks]
            cts.append(ct_common[:, 1:, :].cpu().float())
            torch.cuda.empty_cache()
    outs = torch.cat(outs, dim=0)
    cts = torch.cat(cts, dim=0)

    neuron_activations = outs.view(-1, n_neurons).to(device)
    concept_activations = cts.view(-1, n_neurons).to(device)

    return concept_activations, neuron_activations

def get_onehot_labels(dataset_name, device, superclass_concepts=True):
    """
    Returns:
    onehot_labels: concept activation matrix (|D| x n_concepts)
    concept_text: list of strings with concept names
    """
    concept_set = "data/{}_labels_clean.txt".format(dataset_name.split("_")[0])
    with open(concept_set, 'r') as f: 
        concept_text = (f.read()).split('\n')
    
    dataset = data_utils.get_data(dataset_name)
    num_classes = max(dataset.targets) + 1
    # Convert to one-hot encoding
    onehot_labels = torch.zeros(len(dataset.targets), num_classes)
    onehot_labels[torch.arange(len(dataset.targets)), dataset.targets] = 1
    onehot_labels = onehot_labels.to(device)

    if superclass_concepts:
        with open('data/imagenet_superclass_to_ids.json', 'r') as f:
            superclass_to_id = json.load(f)
        new_labels = []
        for sclass in superclass_to_id.keys():
            concept_text.append(sclass.replace("_", " "))
            subclasses = superclass_to_id[sclass]
            new_labels.append(torch.sum(torch.stack([onehot_labels[:, i] for i in subclasses], dim=0), dim=0))
        new_labels = torch.stack(new_labels, dim=1)
        onehot_labels = torch.cat([onehot_labels, new_labels], dim=1)

    return onehot_labels, concept_text


def save_summary_activations(model, dataset, device, target_layer, batch_size, save_path, pool_mode="avg"):
    act_path = os.path.join(save_path, 'all_{}.pt'.format(pool_mode))
    if os.path.exists(act_path):
        activations = torch.load(act_path, map_location = device)
        return activations.float()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    activations = get_target_activations(model, dataset, [target_layer], batch_size=batch_size, device=device, pool_mode=pool_mode)
    activations = torch.cat(activations[target_layer])
    torch.save(activations.half(), act_path)
    return activations.to(device)

def get_activation_slice(outputs, mode, start=None, end=None):
    '''
    start, end: the endpoints of neurons to record
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output[:, start:end].mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output[:, start:end].detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output[:, start:end].amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output[:, start:end].detach().cpu())
    elif mode=='none':
        def hook(model, input, output):
            outputs.append(output[:, start:end].detach().cpu())
    return hook

def get_target_activations(target_model, dataset, target_layers = ["layer4"], start=None, end=None, batch_size = 128,
                            device = "cuda", pool_mode="none"):
   
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation_slice(all_features[target_layer], pool_mode, start, end))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size, num_workers=8, pin_memory=True):
            features = target_model(images.to(device))
            
    for target_layer in target_layers:
        hooks[target_layer].remove()
            
    return all_features

def save_clip_image_features(model, dataset, save_name, batch_size=256, device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=256):
    if os.path.exists(save_name):
        return torch.load(save_name)
    _make_save_dir(save_name)
    feats = get_clip_text_features(model, text, batch_size)
    torch.save(feats, save_name)
    return feats

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


                                