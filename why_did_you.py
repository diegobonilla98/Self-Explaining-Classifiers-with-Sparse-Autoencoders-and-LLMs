# neuron_topk_for_image.py
import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms, models
import json
import dotenv
dotenv.load_dotenv()
import replicate



neuron_id_to_interpretation = {}
with open(r"neuron_report\llm_output.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        neuron_id = data["neuron_id"]
        answer = data["answer"]
        neuron_id_to_interpretation[neuron_id] = answer

with open("runs\standard_cnn\label_decoder.json", "r", encoding="utf-8") as f:
    label_decoder = json.load(f)

# ---------- Backbone (same as before) ----------
class CamFriendlyResNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        if in_channels == 1:
            w = m.conv1.weight.data
            gray = w.mean(dim=1, keepdim=True)
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad(): m.conv1.weight.copy_(gray)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(512, num_classes)

    def forward(self, x, return_feat: bool = False):
        feat = self.stem(x)                # (B, 512, 7, 7)
        pooled = self.gap(feat).flatten(1) # (B, 512)
        logits = self.fc(pooled)
        return (logits, feat) if return_feat else logits


# ---------- SAE (only need dense encoder) ----------
class TopKSAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.W_e = nn.Parameter(torch.empty(latent_dim, in_dim))
        self.b_e = nn.Parameter(torch.empty(latent_dim))
        self.D   = nn.Parameter(torch.empty(in_dim, latent_dim))
        self.b_d = nn.Parameter(torch.empty(in_dim))

    def encode_dense(self, a):  # (N,C) -> (N,M)
        s = torch.addmm(self.b_e, a, self.W_e.t())
        return torch.relu(s)


# ---------- Helpers ----------
def build_transforms(mean=(0.90,), std=(0.25,)):
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),              # (1,H,W)
        transforms.Normalize(mean, std),
    ])

def load_checkpoints(classifier_ckpt: str, sae_ckpt: str, device: torch.device):
    cstate = torch.load(classifier_ckpt, map_location=device)
    num_classes = int(cstate.get("num_classes", 250))
    mean_img = tuple(cstate.get("mean", (0.90,)))
    std_img  = tuple(cstate.get("std", (0.25,)))
    backbone = CamFriendlyResNet(num_classes=num_classes, in_channels=1, pretrained=False).to(device)
    backbone.load_state_dict(cstate["model"]); backbone.eval()

    sstate = torch.load(sae_ckpt, map_location=device)
    p = sstate["sae_state"]
    if any(k.startswith("module.") for k in p.keys()):
        p = {k.replace("module.",""): v for k,v in p.items()}
    fmean = sstate["feature_mean"].to(device)
    fstd  = sstate["feature_std"].to(device)
    in_dim = fmean.numel()
    latent_dim = p["W_e"].shape[0]
    sae = TopKSAE(in_dim=in_dim, latent_dim=latent_dim).to(device)
    with torch.no_grad():
        sae.W_e.copy_(p["W_e"]); sae.b_e.copy_(p["b_e"])
        sae.D.copy_(p["D"]);     sae.b_d.copy_(p["b_d"])
    sae.eval()
    return backbone, sae, mean_img, std_img, fmean, fstd

@torch.no_grad()
def neuron_scores_for_image(
    pil_or_path: Union[str, Path, Image.Image],
    backbone: CamFriendlyResNet,
    sae: TopKSAE,
    fmean: torch.Tensor,
    fstd: torch.Tensor,
    mean_img=(0.90,), std_img=(0.25,),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    agg: str = "max"  # "max", "mean", or "sum" over 7x7 tokens
) -> Tuple[np.ndarray, torch.Tensor]:
    # Load PIL
    if isinstance(pil_or_path, (str, Path)):
        pil = Image.open(str(pil_or_path)).convert("L")
    else:
        pil = pil_or_path.convert("L")

    tf = build_transforms(mean_img, std_img)
    x = tf(pil).unsqueeze(0).to(device)  # (1,1,224,224)

    with autocast():
        logits, feat = backbone(x, return_feat=True)      # (1, num_classes), (1, C=512, 7, 7)
    B, C, H, W = feat.shape
    a = feat.permute(0,2,3,1).reshape(-1, C)         # (49, 512), unstandardized
    a_std = (a - fmean) / fstd                       # (49, 512)

    z = sae.encode_dense(a_std)                      # (49, M)
    if agg == "max":
        z_img = z.max(dim=0).values                  # (M,)
    elif agg == "mean":
        z_img = z.mean(dim=0)
    elif agg == "sum":
        z_img = z.sum(dim=0)
    else:
        raise ValueError("agg must be one of {'max','mean','sum'}")
    return z_img.float().cpu().numpy(), logits       # (M,), (1, num_classes)

def top_neuron_ids_for_image(
    pil_or_path: Union[str, Path, Image.Image],
    classifier_ckpt: str,
    sae_ckpt: str,
    N: int = 20,
    percentile: float = 99.0,  # keep neurons >= this percentile (within this image)
    agg: str = "max"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, sae, mean_img, std_img, fmean, fstd = load_checkpoints(classifier_ckpt, sae_ckpt, device)
    scores, logits = neuron_scores_for_image(
        pil_or_path, backbone, sae, fmean, fstd, mean_img, std_img, device=device, agg=agg
    )  # (M,), (1, num_classes)

    # Threshold by per-image percentile
    t = np.percentile(scores, percentile)
    mask = scores >= t
    ids = np.flatnonzero(mask)

    # If more than N pass the threshold, take top-N by score
    if ids.size > N:
        order = np.argsort(scores[ids])[::-1][:N]
        ids = ids[order]
    # If fewer than N pass, just return top-N overall
    elif ids.size < N:
        topN = np.argsort(scores)[::-1][:N]
        ids = topN

    # Get predicted class and top 5 predictions
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_class_prob = torch.softmax(logits, dim=1).max().item()
    
    # Get top 5 predictions with probabilities
    probabilities = torch.softmax(logits, dim=1)[0]  # Remove batch dimension
    top5_probs, top5_indices = torch.topk(probabilities, k=5)
    top5_predictions = [(idx.item(), prob.item()) for idx, prob in zip(top5_indices, top5_probs)]
    
    return ids.tolist(), scores[ids].tolist(), float(t), predicted_class, predicted_class_prob, top5_predictions


def call_llm(output_str):
    # The openai/gpt-4.1 model can stream output as it's running.
    chunks = replicate.run(
        "openai/gpt-5",
        input={
            "prompt": output_str,
            "messages": [],
            "image_input": [],
            "system_prompt": "You are an AI explainability agent. You have the task of giving an expanation on why did a neural network classifier output that specific class. For that, you have the output of the top 20 sparse autoencoder neurons, that have been traced back for interpretability, jointly with the activation score (higher the better). Your task is to answer in one sentence to the question \"why did you chose that class?\".\nDon't mention the access to the internal neural activations. Just describe the input characteristics and features that made the model select the given class with the given information.\n\nFor example, an acceptable output can be:\nPrediction: {predicted\\_class} with calibrated probability {p}; I arrived here by detecting the following human-interpretable concepts in the image and combining them with a learned rule set: {primary\\_concepts\\_detected} provided {primary\\_concept\\_contribution}% of the logit evidence, {secondary\\_concepts\\_detected} provided {secondary\\_concept\\_contribution}%, and low-level textures/edges at spatial frequencies {freq\\_range} contributed {texture\\_contribution}%; these were localized mainly in {key\\_regions} (covering {region\\_coverage}% of the object’s area), and ablating these regions drops the class logit by {ablation\\_drop}σ, confirming causality; the evidence was computed via concept-specific circuits {circuit\\_ids} whose behavior has been verified on a held-out causal dataset and passes sufficiency/necessity tests ({sufficiency\\_score}/{necessity\\_score}); counterfactual checks show that replacing {critical\\_attribute} with {counterfactual\\_variant} shifts the top prediction to {alt\\_class} at {alt\\_prob}, while changes to nuisance factors ({lighting}, {background}, {scale}, {pose}) leave the logit within ±{robust\\_margin}; nearest validated prototypes in training are {prototype\\_ids\\_or\\_links} with similarities {sim\\_scores}, and no single example dominated (influence scores ≤ {influence\\_cap}); I rejected close alternatives because {alt1} lacks {missing\\_concept\\_or\\_pattern} (margin {margin1}) and {alt2} exhibits {conflicting\\_cue} inconsistent with {detected\\_concept} (margin {margin2}); the input is in-distribution ({ood\\_score}), passes artifact checks ({artifact\\_checks}), and my uncertainty arises mainly from {ambiguity\\_source}; this explanation is faithful by integrated-gradients infidelity {infidelity} and deletion AUC {deletion\\_auc}, and it is reproducible: re-evaluations under {num\\_augmentations} stochastic augmentations varied the probability by ±{prob\\_std}; training provenance for the supporting concepts traces to {dataset\\_summaries} with balanced coverage across {relevant\\_factors}, and fairness checks indicate no reliance on spurious cues ({spurious\\_cue\\_test}); in plain terms: I see {short\\_human\\_summary\\_of\\_visual\\_evidence}, which is characteristic of a {predicted\\_class}, and removing those cues makes the prediction disappear.\n",
            "verbosity": "medium",
            "reasoning_effort": "medium"
        },
    )
    response = "".join(chunks).strip()
    return response


if __name__ == "__main__":
    # User variables - modify these as needed
    img_path = r"neuron_report\thumbs\feature_00007_img4.webp"  # Path to a grayscale sketch image
    classifier_ckpt = "runs/standard_cnn/best.pt"
    sae_ckpt = "runs/sae_topk/best_sae.pt"
    N = 20
    percentile = 99.0
    agg = "max"  # Choose from: "max", "mean", "sum"

    ids, scores, thr, predicted_class, predicted_class_prob, top5_predictions = top_neuron_ids_for_image(
        img_path, classifier_ckpt, sae_ckpt, N=N, percentile=percentile, agg=agg
    )

    output_str = ""
    output_str += f"Predicted class: \"{label_decoder[str(predicted_class)]}\" with probability {int(predicted_class_prob * 100)}%\n"
    output_str += "\nTop 5 confidence classes:\n"
    for i, (class_idx, prob) in enumerate(top5_predictions):
        class_name = label_decoder[str(class_idx)]
        output_str += f"{i+1}. {class_name}: {prob:.3f} ({int(prob * 100)}%)\n"

    output_str += f"\nPercentile threshold: {thr:.4f}\n"
    for i, (nid, sc) in enumerate(zip(ids, scores)):
        neuron_interpretation = neuron_id_to_interpretation[nid]
        output_str += f"{i+1:02d}. neuron_id={nid:04d}  score={sc:.2f}  {neuron_interpretation}\n"


    print(output_str)
    print("--------------------------------")
    explaination = call_llm(output_str)
    print(explaination)
