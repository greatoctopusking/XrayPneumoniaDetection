import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


def generate_gradcam(model, input_tensor, target_layer=None):
    model.eval()

    if target_layer is None:
        target_layer = model.features[-1]

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor.unsqueeze(0))
    target_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    handle_forward.remove()
    handle_backward.remove()

    act = activations['value'].detach()   # [1, C, H, W]
    grads = gradients['value'].detach()    # [1, C, H, W]

    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * act).sum(dim=1, keepdim=True)   # [1, 1, H, W]
    cam = F.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, target_class


def overlay_heatmap(original_image_path, cam, output_path, alpha=0.5):
    orig = cv2.imread(original_image_path)
    if orig is None:
        return

    h, w = orig.shape[:2]
    heatmap = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(output_path, overlay)
