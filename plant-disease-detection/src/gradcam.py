import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.dataset import get_transforms, IMG_SIZE
from src.model import build_model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device), ckpt["classes"], ckpt["model"]


def target_layer(model, name):
    if name == "resnet50":
        return [model.layer4[-1]]
    return [model.features[-1]]  # mobilenet_v2 / efficientnet_b0


def predict_with_cam(model, name, classes, pil_img, device="cpu"):
    img = pil_img.convert("RGB")
    x = get_transforms(False)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())

    cam = GradCAM(model=model, target_layers=target_layer(model, name))
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]
    rgb = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    return classes[idx], float(probs[idx]), overlay
