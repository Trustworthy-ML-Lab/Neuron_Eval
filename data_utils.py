import os
import torch
import torchvision
import open_clip
from datasets import load_dataset
from transformer_lens import HookedTransformer

from CUB.dataset import CUBDataset

DATASET_ROOTS = {"imagenet_val": "YOUR_PATH/"}

def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {architecture}_{dataset}
                i.e. {resnet18_places365, resnet50_imagenet}
                except for resnet18_places this will return a model trained on ImageNet from torchvision
    """
    if target_name == 'resnet18_places365': 
        target_model = torchvision.models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif "vit_b" in target_name:
        assert ("_imagenet" in target_name)
        target_name = target_name.replace("_imagenet", "")
        target_name_cap = target_name.replace("vit_b", "ViT_B")
        weights = eval("torchvision.models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("torchvision.models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet" in target_name:
        assert ("_imagenet" in target_name)
        target_name = target_name.replace("_imagenet", "")
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif target_name == 'cub_cbm':
        target_model = torch.load('data/cub_cbm_trained.pth', weights_only=False)
        preprocess = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(299),
                torchvision.transforms.ToTensor(), #implicitly divides by 255
                torchvision.transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
    elif target_name == 'cub_linear_probe':
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai",device=device)
        linear = torch.load('data/cub_linear_probe.pth', weights_only=False)
        target_model = ProbeModel(clip_model, linear)
    elif "gpt2" in target_name:
        target_model = HookedTransformer.from_pretrained(target_name).to(device)
        preprocess = None
    target_model.eval()
    return target_model, preprocess

class ProbeModel(torch.nn.Module):
    def __init__(self, clip_backbone, head):
        super(ProbeModel, self).__init__()
        self.clip_backbone = clip_backbone
        self.head = head
    
    def forward(self, x):
        out = self.clip_backbone.encode_image(x)
        out = self.head(out)
        return out

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
                   torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def save_openwebtext_data():
    ds = load_dataset("NeelNanda/openwebtext-tokenized-9b", split="train", streaming=True)
    toks = []
    for i, input in enumerate(ds):
        #print(token)
        if i >= 200:
            break
        toks.append(input["tokens"])
    toks = torch.tensor(toks)
    print(toks.shape, "Saved tokens")
    torch.save(toks, 'data/openweb_tokenized_first_200k.pt')
    return


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = torchvision.datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    elif dataset_name == "places365_val":
        try:
            data = torchvision.datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = torchvision.datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
    elif dataset_name.startswith("cub"): #i.e cub_test
        cub, split = dataset_name.split("_")
        data = CUBDataset([f"data/CUB_processed/class_attr_data_10/{split}.pkl"], use_attr=True, no_img=False,
                               uncertain_label=False, image_dir='images',
                               n_class_attr=2, transform=preprocess)
    elif dataset_name == "openwebtext_subset":
        if not os.path.exists('data/openweb_tokenized_first_200k.pt'):
            save_openwebtext_data()
        data = torch.load('data/openweb_tokenized_first_200k.pt')

    return data