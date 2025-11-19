import os
import glob
import math
import json
from pathlib import Path
import torch
from PIL import Image
from models.dn_dab_deformable_detr.dab_deformable_detr import build_dab_deformable_detr
from util.slconfig import SLConfig
from util import box_ops
import datasets.transforms as T
from draw_box_utils import draw_box

model_config_path = "XX/config.json"
model_checkpoint_path = "XX/checkpoint0199.pth"

# Batch processing
input_dir = "XX/test2017"
output_dir = "XX/vis"   
score_thresh = 0.3
line_thickness = 5

# Category mapping
category_index = {
    1: "benign",
    2: "malignant"
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Load model
def build_model(cfg_path, ckpt_path, device):
    args = SLConfig.fromfile(cfg_path)
    model, criterion, postprocessors = build_dab_deformable_detr(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, postprocessors

def get_transform():
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# inference_batch
def run_inference_batch():
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, postprocessors = build_model(model_config_path, model_checkpoint_path, device)
    transform = get_transform()

    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    img_paths = sorted(img_paths)

    if not img_paths:
        return

    for idx, img_path in enumerate(img_paths, 1):
        try:
            original_img = Image.open(img_path).convert("RGB")
            img_t, _ = transform(original_img, None)
            img_t = img_t.to(device)

            # Forward
            with torch.no_grad():
                output, _ = model(img_t[None], 0)
                H, W = original_img.size[1], original_img.size[0]  # PIL: size=(W,H)
                target_sizes = torch.tensor([[H, W]], dtype=torch.float32, device=device)

                output = postprocessors["bbox"](output, target_sizes)[0]

            scores = output["scores"]              # [N]
            labels = output["labels"]              # [N] 
            boxes_xyxy = output["boxes"]           # [N, 4]  

            select_mask = scores > score_thresh
            if select_mask.sum().item() == 0:
                out_path = os.path.join(output_dir, os.path.basename(img_path))
                original_img.save(out_path)
                continue
            boxes_np   = boxes_xyxy[select_mask].to("cpu").numpy()
            labels_np  = labels[select_mask].to("cpu").numpy()
            scores_np  = scores[select_mask].to("cpu").numpy()
            scores_np = [round(float(s), 2) for s in scores_np]
            draw_box(
                original_img,
                boxes_np,
                labels_np,
                scores_np,
                category_index,
                thresh=score_thresh,
                line_thickness=line_thickness
            )
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            original_img.save(out_path)
        except Exception as e:
            print(f"[ERROR] An error occurred when handling {img_path} : {e}")


if __name__ == "__main__":
    run_inference_batch()



