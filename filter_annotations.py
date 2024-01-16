from collections import defaultdict
import argparse
import glob
import json
import numpy as np
import os
import shutil
from scipy import stats
from tqdm import tqdm
from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb


def filter_annotations(
    unfiltered_data_folder,
    clip_score_key,
    percent_kept,
):
    unfiltered_annotation_path = os.path.join(unfiltered_data_folder, "annotation.json")
    unfiltered_image_folder = os.path.join(unfiltered_data_folder, "images")
    target_folder = unfiltered_data_folder + "_avga%s%d" % (clip_score_key, percent_kept)

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    target_image_folder = os.path.join(target_folder, "images")
    target_annotation_path = os.path.join(target_folder, "annotation.json")

    if not os.path.exists(target_image_folder):
        os.mkdir(target_image_folder)

    with open(unfiltered_annotation_path, "r") as f:
        unfiltered_annotation = json.load(f)
    coco = COCO(unfiltered_annotation_path)

    clip_scores = defaultdict(list)
    for ann in unfiltered_annotation["annotations"]:
        if "segmentation" not in ann:  # only use synthetic data (no GT data) to set the threshold
            clip_scores[ann['category_id']].append(ann[clip_score_key])

    old_images = unfiltered_annotation["images"]
    old_annos = unfiltered_annotation["annotations"]
    new_annos = []
    new_images = []

    # calculate score per image
    all_avg_percentiles = []
    for img_obj in old_images:
        if 'syn' in img_obj.keys():  # take only synthetic images
            annIds = coco.getAnnIds(imgIds=img_obj['id'])
            anns = coco.loadAnns(annIds)
            avg_percentile = 0.
            for ann in anns:
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html
                # stats.percentileofscore([1, 2, 3, 4], 3) -> 75.0
                avg_percentile += stats.percentileofscore(clip_scores[ann['category_id']], ann[clip_score_key])
            if len(anns):
                avg_percentile /= len(anns)
            img_obj["avg_ann_percentile"] = avg_percentile  # the higher the better for objects
            all_avg_percentiles.append(avg_percentile)

    csl_thres = np.percentile(all_avg_percentiles, 100 - percent_kept)

    valid_image_ids = []
    for img_obj in old_images:
        is_valid_image = True
        if 'syn' in img_obj.keys():
            if img_obj["avg_ann_percentile"] < csl_thres:
                is_valid_image = False  # filter out image with score lower than threshold
        if is_valid_image:
            new_images.append(img_obj)
            valid_image_ids.append(img_obj['id'])
            
    valid_ann_ids = coco.getAnnIds(imgIds=valid_image_ids)
    for anno in old_annos:
        if anno['id'] in valid_ann_ids:
            new_annos.append(anno)

    unfiltered_annotation["images"] = new_images
    unfiltered_annotation["annotations"] = new_annos

    with open(target_annotation_path, "w+") as f:
        json.dump(unfiltered_annotation, f)

    img_names = [img['file_name'] for img in unfiltered_annotation["images"]]
    for img_name in img_names:
        shutil.copy(os.path.join(unfiltered_image_folder, img_name), os.path.join(target_image_folder, img_name))

    print(f"num old_images: {len(old_images)}")
    print(f"num old_annos: {len(old_annos)}")
    print(f"num new_images: {len(new_images)}")
    print(f"num new_annos: {len(new_annos)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--unfiltered_data_folder", default=None, type=str)
    parser.add_argument("-k", "--clip_score_key", default=None, type=str)
    parser.add_argument("-p", "--percent_kept", default=None, type=int)
    args = parser.parse_args()

    '''
    e.g. coco10novel (coco 10 shot, novel cat only)
    python3 5av1_filter_based_on_clip_rank.py \
        -d /media/data/ControlAug/cnet/experiments/coco10s1_512p/mix_n333-333_dfsNone_o0_m0_s1_HED_p512_promptcat_seed1_noprompt_imprior \
        -k csl \
        -p 30
    '''

    filter_annotations(
            unfiltered_data_folder=args.unfiltered_data_folder,
            clip_score_key=args.clip_score_key,
            percent_kept=args.percent_kept,
        )


if __name__ == "__main__":
    main()
