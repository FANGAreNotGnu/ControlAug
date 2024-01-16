import argparse
import glob
import json
import numpy as np
import os
import pickle
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
from PIL import Image

FOLDER_PREFIXS=["JPEGImages/","images/apple/TRAIN_RGB/","images/avocado/TRAIN_RGB/",
                "images/capsicum/TRAIN_RGB/","images/mango/TRAIN_RGB/","images/orange/TRAIN_RGB/",
                "images/rockmelon/TRAIN_RGB/","images/strawberry/TRAIN_RGB/","images/"]

def get_image_sample_name(syn_data_path, clip_score_key, filter_strategy):
    if clip_score_key is None:
        if filter_strategy not in ["nofilter"]:
            raise ValueError("clip_score_key should not be None")  # TODO: could add default behavior (e.g. return syn001.jpg)
    if filter_strategy is None:
        raise ValueError("filter_strategy should not be None")  # TODO: could add default behavior (e.g. return syn001.jpg)

    if filter_strategy == "best_mix_scores":
        scores_path = os.path.join(syn_data_path, "mix_scores.pickle")
        if not os.path.exists(scores_path):
            return None  # the data sample not valid and thus skipped
        with open(scores_path, "rb") as f:
            scores = pickle.load(f)

        best_idx = np.argmax(scores[clip_score_key])
    elif filter_strategy == "best_ann_scores":
        scores_path = os.path.join(syn_data_path, "ann_scores.pickle")
        if not os.path.exists(scores_path):
            return None  # the data sample not valid and thus skipped
        with open(scores_path, "rb") as f:
            scores = pickle.load(f)

        best_idx = np.argmax(scores[clip_score_key])
    elif filter_strategy == "best_bg_scores":
        scores_path = os.path.join(syn_data_path, "bg_scores.pickle")
        if not os.path.exists(scores_path):
            return None  # the data sample not valid and thus skipped
        with open(scores_path, "rb") as f:
            scores = pickle.load(f)

        best_idx = np.argmin(scores[clip_score_key])
    elif filter_strategy == "two_stage":
        ann_scores_path = os.path.join(syn_data_path, "ann_scores.pickle")
        if not os.path.exists(ann_scores_path):
            return None  # the data sample not valid and thus skipped
        with open(ann_scores_path, "rb") as f:
            ann_scores = pickle.load(f)

        bg_scores_path = os.path.join(syn_data_path, "bg_scores.pickle")
        if not os.path.exists(bg_scores_path):
            return None  # the data sample not valid and thus skipped
        with open(bg_scores_path, "rb") as f:
            bg_scores = pickle.load(f)

        N = len(ann_scores[clip_score_key])
        first_stage_N = int(N ** .5)
        first_stage_idx = np.argsort(ann_scores[clip_score_key])[::-1][:first_stage_N]
        best_idx = np.argmin(np.array(bg_scores[clip_score_key])[first_stage_idx])
        best_idx = first_stage_idx[best_idx]
    elif filter_strategy == "nofilter":
        best_idx = 0
    else:
        raise ValueError(f"filter_strategy: {filter_strategy}")
    
    return "syn%03d.jpg" % best_idx

def mix_annotations(
            gt_annotation_path,
            gt_image_folder,
            syn_annotation_folders,
            target_folder,
            filter_strategy,
            clip_score_key,
            num_synthetic_samples,
        ):

    if target_folder is None:
        raise ValueError(f"target_folder not provided, but it is required when multiple syn datasets are mixed")

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    target_image_folder = os.path.join(target_folder, "images")
    target_annotation_path = os.path.join(target_folder, "annotation.json")

    if not os.path.exists(target_image_folder):
        os.mkdir(target_image_folder)

    with open(gt_annotation_path, "r") as f:
        source_annotation = json.load(f)
    coco = COCO(gt_annotation_path)

    # 1. copy source image data
    img_names = [img['file_name'] for img in coco.loadImgs(coco.getImgIds())]
    for img_name in img_names:
        target_image_name = img_name
        for folder_prefix in FOLDER_PREFIXS:
            target_image_name = target_image_name[len(folder_prefix):] if target_image_name[:len(folder_prefix)] == folder_prefix else target_image_name
        shutil.copy(os.path.join(gt_image_folder, img_name), os.path.join(target_image_folder, target_image_name))

    print(f"num gt images: {coco.getImgIds()}")
    print(f"num gt annotations: {coco.getAnnIds()}")
    print(f"num gt categories: {coco.getCatIds()}")

    # 2. add sync anno to annotation, copy syn image
    curr_img_id = max(coco.getImgIds()) + 1
    curr_ann_id = max(coco.getAnnIds()) + 1

    max_img_id = curr_img_id + num_synthetic_samples if num_synthetic_samples is not None else curr_img_id + 99999999

    for syn_annotation_folder in syn_annotation_folders:
        syn_data_paths = sorted(glob.glob(os.path.join(syn_annotation_folder, "*")))
        new_images = source_annotation["images"]
        # remove additional folder prefix
        for i, img_obj in enumerate(new_images):
            f_name = img_obj['file_name']
            for folder_prefix in FOLDER_PREFIXS:
                f_name = f_name[len(folder_prefix):] if f_name[:len(folder_prefix)] == folder_prefix else f_name
            new_images[i]['file_name'] = f_name
        new_anns = source_annotation["annotations"]

        for syn_data_path in tqdm(syn_data_paths):
            image_sample_name = get_image_sample_name(syn_data_path, clip_score_key, filter_strategy)
            if image_sample_name is None:
                continue

            # copy sync image
            syn_image_path = os.path.join(syn_data_path, image_sample_name)  # source path  # TODO: only support sample = 1
            img = Image.open(syn_image_path)
            # get width and height
            width = img.width
            height = img.height

            syn_image_name = "%012d.jpg"%curr_img_id  # target name
            shutil.copy(syn_image_path, os.path.join(target_image_folder, syn_image_name))
            
            layout_cats_path = os.path.join(syn_data_path, "layout_cats.npy")
            layout_bboxes_path = os.path.join(syn_data_path, "layout_bboxes.npy")
            prompt_path = os.path.join(syn_data_path, "prompt.npy")
            cats = np.load(layout_cats_path)
            bboxes = np.load(layout_bboxes_path)
            if os.path.exists(prompt_path):
                prompt = np.load(prompt_path)[0]
            else:
                prompt = ""  # for PbE, prompt is an example image
            
            catids = [coco.getCatIds(catNms=[cat])[0] for cat in cats]
            
            # add sync anno to annotation
            # TODO: image shape hard coded
            new_images.append({'file_name': syn_image_name, 'height': height, 'width': width, 'id': curr_img_id, 'prompt': prompt, "syn": True,})
            
            num_objects = cats.shape[0]
            for i in range(num_objects):
                area = float(bboxes[i][-1] * bboxes[i][-2])
                if area < 10:
                    continue
                new_anns.append({
                    'image_id': curr_img_id, 
                    'bbox': bboxes[i].tolist(), 
                    'area': area, 
                    'category_id': catids[i], 
                    'id': curr_ann_id,
                    "syn": True,
                })
                curr_ann_id += 1
            
            curr_img_id += 1
            #print(syn_image_path)
            if curr_img_id >= max_img_id:
                print(f"reach max_img_id, stop adding synthetic images")
                break # will add one more image in rest syn_annotation_folders even it reached the limit, but it's ok

    source_annotation["images"] = new_images
    source_annotation["annotations"] = new_anns

    with open(target_annotation_path, "w+") as f:
        json.dump(source_annotation, f)

    print(f"num images: {len(source_annotation['images'])}")
    print(f"num annotations: {len(source_annotation['annotations'])}")
    print(f"num categories: {len(source_annotation['categories'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--gt_annotation_path", default=None, type=str)
    parser.add_argument("--gt_image_folder", default="/media/data/coco_fsod/train2017", type=str)
    parser.add_argument("-s", "--syn_annotation_folders", nargs='+', required=True)
    parser.add_argument("-t", "--target_folder", type=str, required=True)
    parser.add_argument("-f", "--filter_strategy", type=str, default=None)
    parser.add_argument("-k", "--clip_score_key", type=str, default=None)
    parser.add_argument("-n", "--num_synthetic_samples", type=int, default=None)
    args = parser.parse_args()

    '''
    python3 4v1b_mix_annotations.py \
        -a /media/data/coco_fsod/seed1/10shot_novel.json \
        -s /media/data/ControlAug/cnet/experiments/coco10s1_512p/syn_n2000_o0_m0_s1_p512_pbefixed \
        -t /media/data/ControlAug/cnet/experiments/coco10s1_512p/mix_n2000-167_o0_m0_s1_p512_pbefixed \
        -f nofilter \
        -n 167


    python3 4v1b_mix_annotations.py \
        -a /media/data/coco_fsod/seed1/10shot_novel.json \
        -s /media/data/ControlAug/cnet/experiments/coco10s1_512p/syn_n200_o0_m0_s25_canny_p512_imprior \
        -t /media/data/ControlAug/cnet/experiments/coco10s1_512p/mix_n200_o0_m0_s25_canny_p512_imprior_cslp_twostage \
        -f two_stage \
        -k csl_p \
        -n 1000
        
    '''

    mix_annotations(
            gt_annotation_path=args.gt_annotation_path,
            gt_image_folder=args.gt_image_folder,
            syn_annotation_folders=args.syn_annotation_folders,
            target_folder=args.target_folder,
            filter_strategy=args.filter_strategy,
            clip_score_key=args.clip_score_key,
            num_synthetic_samples=args.num_synthetic_samples,
        )


if __name__ == "__main__":
    main()

