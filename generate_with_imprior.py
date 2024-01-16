from tqdm import tqdm
import argparse
import einops
# import gradio as gr
import numpy as np
import torch
import os
import random

from vis_prior.utils import *
from vis_prior.vis_prior_generator import CannyVPG, HEDVPG, UniformerVPG, ScribbleVPG

import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../ControlNet"))
from share import *
import config
from pytorch_lightning import seed_everything
from annotator.util import HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.plms_hacked import PLMSSampler


def process_hed(model, ddim_sampler, vis_prior, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        detected_map = HWC3(vis_prior.astype(np.uint8))
        H, W, C = detected_map.shape

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed is not None:
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


def generate_layout_and_synthetic_images(
        annotation_path, 
        im_folder, 
        model_config_path, 
        experiment_path, 
        ckpt_name, 
        num_layouts, 
        num_samples_per_layout, 
        pixels_size,
        diffusion_seed,
        synthetic_data_postfix = None,
        vpg_mode = "HED",
        prompt_mode = "cat",
        ckpt_path = None,
        morphology_kernal_size=0,
        postfix2=None,
        seed = None,
    ):
    # 0. Setting up save path and checkpoint path
    synthetic_data_save_name = f"syn_n{num_layouts}{'_s' + str(num_samples_per_layout) if num_samples_per_layout != 1 else ''}{'_m' + str(morphology_kernal_size) if morphology_kernal_size else ''}_{vpg_mode}_p{pixels_size}_pr{prompt_mode}_dfs{diffusion_seed if diffusion_seed is not None else 'None'}_seed{seed if seed is not None else 'None'}{'_'+synthetic_data_postfix if synthetic_data_postfix is not None else ''}{'_'+postfix2 if postfix2 is not None else ''}"
    synthetic_data_save_path = os.path.join(experiment_path, synthetic_data_save_name)
    if not os.path.exists(synthetic_data_save_path):
        os.makedirs(synthetic_data_save_path)
    if ckpt_path is None:
        ckpt_path = os.path.join(experiment_path, ckpt_name)

    # 1. Initialize Visual Prior Generator (vpg)
    vpg_params = {"vpl": None,  # do not need layout generator for image prior
                  "fill_val": 0,
                  "annotation": annotation_path,
                  "im_folder": im_folder,
                  "mode": "image",
                  "prompt_mode": prompt_mode,
                  }
    if vpg_mode == "HED":
        vpg = HEDVPG(vpg_params=vpg_params) 
    elif vpg_mode == "scribble":
        vpg = ScribbleVPG(vpg_params=vpg_params) 
    elif vpg_mode == "canny":
        detector_params = {"low": 100,
                           "high": 200,
                           "blur": 0,
                           }
        vpg = CannyVPG(vpg_params=vpg_params, detector_params=detector_params)
    elif vpg_mode == "uniformer":
        vpg = UniformerVPG(vpg_params=vpg_params) 
    else:
        raise ValueError(f"vpg_mode: {vpg_mode}")

    # 2. Generate visual priors (together with other ControlNet inputs)
    image_visual_priors = vpg.sample_image_visual_priors(num_layouts=num_layouts, pixel_size=pixels_size, morphology_kernal_size=morphology_kernal_size)  # TODO: add to args
    prompts = [image_visual_prior["prompts"][0] for image_visual_prior in image_visual_priors]
    vis_priors = [image_visual_prior["vis_priors"][0] for image_visual_prior in image_visual_priors]

    # 3. Initialize ControlNet
    model = create_model(model_config_path).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # 4. Generate and save images
    for i in tqdm(range(num_layouts)):
        results_per_layout = [process_hed(
                        model=model,
                        ddim_sampler=ddim_sampler,
                        vis_prior=vis_priors[i],
                        #prompt="picture of a " + prompts[i],
                        prompt=prompts[i],
                        #a_prompt="realistic, real, photo", 
                        #a_prompt="cinematic, colorful background, concept art, dramatic lighting, high detail, highly detailed, hyper realistic, intricate, intricate sharp details, octane render, smooth, studio lighting, trending on artstation", 
                        #a_prompt="highly detailed, sharp focus, concept art, intricate, artstation, digital painting, smooth, elegant, illustration, cinematic lighting, octane render, trending on artstation, 8 k, dramatic lighting, cinematic",
                        a_prompt="",
                        #n_prompt="blurry, unclear, duplicate, distortion, lowres, bad anatomy, error, cropped, low quality", 
                        n_prompt="", 
                        num_samples=1, 
                        ddim_steps=50, 
                        guess_mode=False, 
                        strength=1., 
                        scale=9, 
                        seed=diffusion_seed, 
                        eta=0.,
                    )[1] for j in range(num_samples_per_layout)]
        #results.append(results_per_layout)

    # results: a (num_layouts, num_samples_per_layout) list of images
    # layouts: a (num_layouts,) list of layout
    # layout: [[category_name, bbox, (im_prior)], ...]
    #for i in range(num_layouts):
        synthetic_sample_save_path = os.path.join(synthetic_data_save_path, "%08d"%i)
        if not os.path.exists(synthetic_sample_save_path):
            os.mkdir(synthetic_sample_save_path)

        image_visual_prior = image_visual_priors[i]
        layout_cats_save_path = os.path.join(synthetic_sample_save_path, "layout_cats.npy")
        layout_bboxes_save_path = os.path.join(synthetic_sample_save_path, "layout_bboxes.npy")
        layout_priors_save_path = os.path.join(synthetic_sample_save_path, "layout_priors.npy")

        cats = image_visual_prior["cats"]
        bboxes = image_visual_prior["bboxes"]
        priors = image_visual_prior["vis_priors"]

        np.save(layout_cats_save_path, cats)
        np.save(layout_bboxes_save_path, bboxes)
        np.save(layout_priors_save_path, priors)

        # write vis_prior for this layout
        imwrite(os.path.join(synthetic_sample_save_path, "vis_prior.jpg"), vis_priors[i])

        # write synthetic images for this layout
        for j in range(num_samples_per_layout):
            imwrite(os.path.join(synthetic_sample_save_path, "syn%03d.jpg"%j), results_per_layout[j])

        # write prompt for this layout TODO: we'll have multiple prompt per sample
        prompt = prompts[i]
        prompt_save_path = os.path.join(synthetic_sample_save_path, "prompt.npy")
        np.save(prompt_save_path, [prompt])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation_path", default=None, type=str)
    parser.add_argument("-i", "--im_folder", default=None, type=str)
    parser.add_argument("--model_config_path", default='/media/wacv/ControlNet/models/cldm_v15.yaml', type=str)
    parser.add_argument("-e", "--experiment_path", default=None, type=str)
    parser.add_argument("-c", "--ckpt_name", default=None, type=str)
    parser.add_argument("-l", "--num_layouts", default=None, type=int)
    parser.add_argument("-s", "--num_samples_per_layout", default=None, type=int)
    parser.add_argument("-p", "--pixels_size", default=None, type=int)  # now we generate square images only
    parser.add_argument("--synthetic_data_postfix", default="imprior", type=str)  # default prefix for using image as prior
    parser.add_argument("--vpg_mode", default="HED", type=str)
    parser.add_argument("--prompt_mode", default="cat", type=str)
    parser.add_argument("--ckpt_path", default='/media/wacv/ControlNet/models/control_sd15_hed.pth', type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--diffusion_seed", default=None, type=int)
    parser.add_argument("-m", "--morphology_kernal_size", default=0, type=int)  # now we generate square images only
    parser.add_argument("--postfix2", default=None, type=str)
    args = parser.parse_args()

    set_seed(args.seed)

    '''
    e.g. coco10s1_512p
    CUDA_VISIBLE_DEVICES=1 python3 3v1_generate_synthetic_images_wiith_same_prior.py \
        -a /media/data/coco_fsod/seed1/10shot_novel.json \
        -i /media/data/coco_fsod/train2017/ \
        -e /media/data/ControlAug/cnet/experiments/coco10s1_512p \
        -l 2 \
        -s 1 \
        -p 512 \
        --seed 1
    '''

    generate_layout_and_synthetic_images(
        annotation_path=args.annotation_path, 
        im_folder=args.im_folder, 
        model_config_path=args.model_config_path, 
        experiment_path=args.experiment_path, 
        ckpt_name=args.ckpt_name, 
        num_layouts=args.num_layouts, 
        num_samples_per_layout=args.num_samples_per_layout, 
        pixels_size=args.pixels_size, 
        synthetic_data_postfix=args.synthetic_data_postfix,
        vpg_mode=args.vpg_mode,
        ckpt_path=args.ckpt_path,
        morphology_kernal_size=args.morphology_kernal_size,
        diffusion_seed=args.diffusion_seed,
        postfix2=args.postfix2,
        prompt_mode=args.prompt_mode,
        seed=args.seed,
    )   

if __name__ == "__main__":
    main()
