import logging
import os

import open_clip
import torch
from tqdm import tqdm

from .data.clip_datasets.registry import get_dataset
from .data.clip_datasets.templates import get_templates
from .modeling import ClassificationHead, ImageEncoder

log = logging.getLogger(__name__)


def build_classification_head(model, dataset_name, template, data_location, device):
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(dataset_name, None, location=data_location)
    model.eval()
    model.to(device)

    log.info("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(
    cfg, dataset: str, freeze: bool = True
) -> ClassificationHead:
    """
    Gets the classification head for the given model and dataset.

    Args:
        cfg : configuration
        dataset (str): The name of the dataset.
        freeze (bool): Whether to freeze the weights of the classification head.

    Returns:
        ClassificationHead: The classification head for the given model and dataset.
    """
    filename = os.path.join(cfg.save, f"head_{dataset}.pt")
    if os.path.exists(filename):  # file exists
        log.info(
            f"Classification head for {cfg.model} on {dataset} exists at {filename}"
        )
        head = ClassificationHead.load(filename)
    else:  # file does not exist, build one from scratch
        log.info(
            f"Did not find classification head for {cfg.model} on {dataset} at {filename}, building one from scratch."
        )
        model = ImageEncoder(cfg, keep_lang=True).model
        template = get_templates(dataset)
        classification_head = build_classification_head(
            model, dataset, template, cfg.data_location, cfg.device
        )
        os.makedirs(cfg.save, exist_ok=True)
        classification_head.save(filename)
        head = classification_head
    if freeze:
        head.requires_grad_(False)
    return head
