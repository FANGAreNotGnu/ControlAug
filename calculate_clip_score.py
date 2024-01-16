import argparse
import glob
import json
import numpy as np
import os
import shutil


import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
import clip

from constants import PROMPT_ENGINEERING
from vis_prior.utils import *


def calculate_ann_clip_score(
    unfiltered_data_folder,
    clip_mode,
    device,
):

    assert clip_mode in ["l", "b"]
    if clip_mode == "l":
        clip_model_name =  "ViT-L/14"
    elif clip_mode == "b":
        clip_model_name =  "ViT-B/32"
    else:
        raise ValueError(f"clip_mode should be in {['l', 'b']}, but it is: {clip_mode}")

    unfiltered_annotation_path = os.path.join(unfiltered_data_folder, "annotation.json")

    with open(unfiltered_annotation_path, "r") as f:
        unfiltered_annotation = json.load(f)
    coco = COCO(unfiltered_annotation_path)

    model, preprocess = clip.load(clip_model_name, device=device)

    for ann in tqdm(unfiltered_annotation['annotations']):
        ann_id = ann["id"]
        x, y, w, h = ann['bbox']
        
        image_id = ann['image_id']
        img_obj = coco.loadImgs([image_id])[0]
        img_path = os.path.join(unfiltered_data_folder, 'images', img_obj['file_name'])
        img = Image.open(img_path).convert("RGB")
        img = img.crop((x, y, x+w, y+h))
        #img = imread(img_path)
        
        cat_id = ann['category_id']
        cat_obj = coco.loadCats([cat_id])[0]
        cat_name = cat_obj["name"]
        
        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize([pe(cat_name) for k, pe in PROMPT_ENGINEERING[clip_mode].items()]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
        
        for i, (k, pe) in enumerate(PROMPT_ENGINEERING[clip_mode].items()):
            ann[k] = float(logits_per_image[0][i]/100)

    with open(unfiltered_annotation_path, "w+") as f:
        json.dump(unfiltered_annotation, f)

    print(len(unfiltered_annotation['annotations']))
    print(unfiltered_annotation['annotations'][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--unfiltered_data_folder", default=None, type=str)
    parser.add_argument("--clip_mode", default="l", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    '''
    e.g. coco10novel (coco 10 shot, novel cat only)
    CUDA_VISIBLE_DEVICES=0 python3 5_calculate_ann_clip_score.py \
        -d /media/data/ControlAug/cnet/experiments/coco10s1_512p/mix_n333-333_o0_m0_s1_HED_p512_imprior_r1
    '''

    calculate_ann_clip_score(
            unfiltered_data_folder=args.unfiltered_data_folder,
            clip_mode=args.clip_mode,
            device=args.device,
        )


if __name__ == "__main__":
    main()
