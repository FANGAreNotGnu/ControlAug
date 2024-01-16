import cv2 as cv
import numpy as np
import os
import random
import torch
from collections import defaultdict

from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

from .prompt_constructor import PromptConstructor
from .utils import crop_bboxes, imread, random_crop_and_resize_images_to_square, copy_numpy_to_pil, copy_pil_to_numpy, get_bboxes_mask


VALID_PRMOPT_MODES = ["cat", "and", "set", "setand", "shuffledset", "shuffledsetand", "img", "pic", "mix"]

class VisPriorGenerator():
    def __init__(self, 
                 fill_val, 
                 detector, 
                 vpl=None,  # for bbox layout only
                 sample_channel=None, 
                 annotation=None, 
                 im_folder=None, 
                 mode="bbox", 
                 prompt_mode="cat",
                 valid_prompt_modes=None,
                 ):
        # mode: (visual_prior_mode)
        #   - bbox: use visual priors in bboxes
        #   - image: use visual prior of whole image
        #   - syn: use synthetic visual priors
        #
        # prompt_mode:
        #   - concat: concat categories
        #   - blipv2: use blipv2 from image to text

        self.vpl = vpl
        self.detector = detector
        self.fill_val = fill_val
        self.sample_channel = sample_channel

        self.prior_bank = defaultdict(list)
        self.annotation = annotation
        self.im_folder = im_folder
        self.mode = mode
        self.prompt_mode = prompt_mode
        self.valid_prompt_modes = valid_prompt_modes

        self.prompt_constructor = PromptConstructor(prompt_mode=self.prompt_mode, valid_prompt_modes=self.valid_prompt_modes)

        if self.annotation is not None and im_folder is not None:
            if self.mode == "bbox":
                self.update_prior_bank_with_bbox(self.annotation, self.im_folder)
            elif self.mode == "image":
                self.update_prior_bank_with_images(self.annotation, self.im_folder)
            elif self.mode == "syn":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown VPG mode: {self.mode}")

    def update_prior_bank_with_bbox(self, annotation, im_folder):
        assert isinstance(annotation, str), f"annotation should be a str (json's path), but it is {type(annotation)}"
        coco = COCO(annotation)
        #if isinstance(annotation, str):
        #    with open(annotation, "r") as f:
        #        annos = json.load(f)
        #elif isinstance(annotation, dict):
        #    annos = annotation
        #else:
        #    raise ValueError(f"annotation should be a str (json's path) or a dict, but it is {type(annotation)}")
        annIds = coco.getAnnIds()
        anns = coco.loadAnns(annIds)
        for ann in anns:
            try:
                image_id = ann["image_id"]
                img = coco.loadImgs([image_id])[0]

                cat_id = ann["category_id"]
                cat = coco.loadCats([cat_id])[0]

                image_file_name = img['file_name']
                category_name = cat['name']

                img = imread(os.path.join(im_folder, image_file_name))
                bbox = ann['bbox']

                vis_prior = self.detect_one_bbox(img, bbox)

                self.prior_bank[category_name].append(vis_prior)

            except Exception as e:
                print(e)  # Output size is too small is due to small boundingbox
                print(f"failed to detect annotation: {ann['id']}")

        print("prior bank updated with bbox priors")

    def update_prior_bank_with_images(self, annotation, im_folder):
        assert isinstance(annotation, str), f"annotation should be a str (json's path), but it is {type(annotation)}"
        coco = COCO(annotation)
        #if isinstance(annotation, str):
        #    with open(annotation, "r") as f:
        #        annos = json.load(f)
        #elif isinstance(annotation, dict):
        #    annos = annotation
        #else:
        #    raise ValueError(f"annotation should be a str (json's path) or a dict, but it is {type(annotation)}")
        imgIds = coco.getImgIds()
        for img_obj in coco.loadImgs(imgIds):
            image_file_name = img_obj['file_name']
            image_id = img_obj["id"]
            img = imread(os.path.join(im_folder, image_file_name))

            annIds = coco.getAnnIds(imgIds=[image_id])
            anns = coco.loadAnns(annIds)
            bboxes = [ann["bbox"] for ann in anns]

            catIds = [ann["category_id"] for ann in anns]
            cats = coco.loadCats(catIds)
            cat_names = [cat["name"] for cat in cats]

            vis_prior = self.detect_one_img(img)

            image_visual_prior = {}
            image_visual_prior["vis_priors"] = vis_prior[None, :, :, :]
            image_visual_prior["img"] = img[None, :, :, :]
            image_visual_prior["cats"] = cat_names 
            image_visual_prior["prompts"] = self.prompt_constructor(cats=cat_names, img=img)
            image_visual_prior["bboxes"] = bboxes 

            self.prior_bank["image_visual_priors"].append(image_visual_prior)

        print("prior bank updated with image priors")

    def generate_layouts(self, im_shape, num_object_per_layout, num_layouts):
        layouts = []
        for i in range(num_layouts):
            layouts.append(
                self.vpl.generate_a_layout_with_prior(
                        im_shape=im_shape, 
                        priors=self.prior_bank, 
                        num_object=num_object_per_layout,
                    )
            )
        
        return layouts

    def generate_prompts(self, layouts):
        return [", ".join([item[0] for item in layout]) for layout in layouts]  # concat all category names

    def generate_vis_priors(self, layouts, im_shape, fill_val=None,):
        vis_priors = []
        assert im_shape[2] == self.sample_channel, "Input im_shape channel error!"
        for layout in layouts:
            vis_prior = self.draw_one_layout(im_shape=im_shape, layout=layout, fill_val=fill_val)
            vis_priors.append(vis_prior)

        return vis_priors
    
    def sample_image_visual_priors(self, num_layouts, pixel_size, morphology_kernal_size=0):
        if self.prompt_constructor is None:
            self.prompt_constructor = PromptConstructor(prompt_mode=self.prompt_mode,
                                                        valid_prompt_modes=self.valid_prompt_modes)

        prior_indices = random.choices(range(len(self.prior_bank["image_visual_priors"])), k=num_layouts)
        image_visual_priors = []
        for i in prior_indices:
            img_visual_prior = self.prior_bank["image_visual_priors"][i]
            vis_prior, bboxes, cat_names = random_crop_and_resize_images_to_square(imgs=img_visual_prior["vis_priors"], square_size=pixel_size, bboxes=img_visual_prior["bboxes"], cats=img_visual_prior["cats"])

            if morphology_kernal_size:
                kernel = np.ones((morphology_kernal_size, morphology_kernal_size), np.uint8)
                type_of_morph = random.randint(0, 4)
                if type_of_morph == 0:
                    pass
                elif type_of_morph == 1:
                    vis_prior[0] = cv.erode(vis_prior[0], kernel, iterations=1)[:,:,None]
                elif type_of_morph == 2:
                    vis_prior[0] = cv.dilate(vis_prior[0], kernel, iterations=1)[:,:,None]
                elif type_of_morph == 3:
                    vis_prior[0] = cv.erode(vis_prior[0], kernel, iterations=1)[:,:,None]
                    vis_prior[0] = cv.dilate(vis_prior[0], kernel, iterations=1)[:,:,None]
                elif type_of_morph == 4:
                    vis_prior[0] = cv.dilate(vis_prior[0], kernel, iterations=1)[:,:,None]
                    vis_prior[0] = cv.erode(vis_prior[0], kernel, iterations=1)[:,:,None]   

            new_img_visual_prior = {}
            new_img_visual_prior["vis_priors"] = vis_prior
            new_img_visual_prior["cats"] = cat_names
            new_img_visual_prior["prompts"] = self.prompt_constructor(cats=cat_names, img=img_visual_prior["img"][0])
            new_img_visual_prior["bboxes"] = bboxes 

            image_visual_priors.append(new_img_visual_prior)

        # clean GPU
        self.prompt_constructor = None

        return image_visual_priors

    def detect_one_bbox(self, img, bbox):
        x, y, w, h = bbox

        im_prior = self.detector(img=img.copy())
        if im_prior.ndim == 2:
            im_prior = im_prior[:, :, None]
        bbox_prior = im_prior[int(y):int(y+h), int(x):int(x+w), :]

        return bbox_prior

    def detect_one_img(self, img, bboxes=None, detector=None):
        if detector is None:
            detector = self.detector

        im_prior = detector(img=img)
        if im_prior.ndim == 2:
            im_prior = im_prior[:, :, None]

        if bboxes is not None and self.mode=="bbox":
            im_prior = crop_bboxes(img=im_prior, bboxes=bboxes)

        return im_prior

    def draw_one_bbox(self, im_prior, new_bbox, fill_val=None, target_img=None, im_shape=None):
        x_new, y_new, w_new, h_new = new_bbox

        if fill_val is None:
            fill_val = self.fill_val
        
        # init target_img to draw if not provided
        if target_img is None:
            assert im_shape is not None, "Failed to initialize the target image! Note that target_img and im_shape can not both be None."
            target_img = np.zeros((im_shape[0],im_shape[1],im_prior.shape[2]))
            target_img.fill(fill_val)
        else:
            assert target_img.shape[0] == im_shape[0], f"shape miss match: {target_img.shape[0]} != {im_shape[0]}"
            assert target_img.shape[1] == im_shape[1], f"shape miss match: {target_img.shape[1]} != {im_shape[1]}"
            assert target_img.shape[2] == self.sample_channel, f"shape miss match: {target_img.shape[2]} != {self.sample_channel}"

        im_prior = cv.resize(im_prior, dsize=(w_new, h_new))
        if im_prior.ndim == 2:  # cv resize will eliminate last dimension if it's 1
            im_prior = im_prior[:, :, None]

        target_img[int(y_new):int(y_new)+im_prior.shape[0], int(x_new):int(x_new)+im_prior.shape[1], :] = im_prior  # TODO: consider add (with transparency?) to previous bbox when we use multiple bbox in an image (currently overwrite)

        return target_img

    def draw_one_layout(self, im_shape, layout, fill_val=None):
        # layout (without prior): [[category1, bbox1], [category2, bbox2], ...]
        # layout (with prior): [[category1, bbox1, prior1], [category2, bbox2, prior2], ...]
        # layout is ordered
        # im_shape: H, W, C

        if fill_val is None:
            fill_val = self.fill_val

        target_img = np.zeros(im_shape)
        target_img.fill(fill_val)

        for layout_object in layout:
            if len(layout_object) == 2:
                category_name, bbox = layout_object
                im_prior = self.sample_im_prior(category_name)  #TODO
            elif len(layout_object) == 3:
                category_name, bbox, im_prior = layout_object
            else:
                raise ValueError(f"the length of layout_object should be 2 or 3, but is {len(layout_object)}")

            target_img = self.draw_one_bbox(im_prior=im_prior, new_bbox=bbox, fill_val=fill_val, target_img=target_img, im_shape=im_shape)

        return target_img

    def visualize_one_bbox(self, img, bbox, category_name=None, prior_mode=None, fill_val=None, target_img=None):
        # used for simple visualization
        # img: source img

        new_bbox = self.vpl.generate_bbox(im_shape=img.shape, prior_shape=(bbox[3],bbox[2]))  # prior shape=hw

        im_prior = self.detect_one_bbox(img, bbox)

        target_img = self.draw_one_bbox(im_prior, new_bbox, fill_val=fill_val, target_img=target_img, im_shape=img.shape if target_img is None else target_img.shape)

        return target_img 

    def visualize_one_layout(self, im_shape, category_name=None, prior_mode=None, fill_val=None,):
        # used for simple visualization

        layout = self.vpl.generate_a_layout_with_prior(im_shape=im_shape, priors=self.prior_bank, num_object=3)  # prior shape=hw

        target_img = self.draw_one_layout(fill_val=fill_val, im_shape=(im_shape[0], im_shape[1], self.sample_channel), layout=layout)

        return target_img 


class CannyVPG(VisPriorGenerator):
    def __init__(self, vpg_params, detector_params):
        from .vis_prior_detector import CannyEdgeDetector
        detector = CannyEdgeDetector(**detector_params)
        super().__init__(detector=detector, **vpg_params)


class HEDVPG(VisPriorGenerator):
    def __init__(self, vpg_params, detector_params=None):
        from .vis_prior_detector import HEDEdgeDetector
        detector = HEDEdgeDetector()
        super().__init__(detector=detector, **vpg_params)


class ScribbleVPG(VisPriorGenerator):
    def __init__(self, vpg_params, detector_params=None):
        from .vis_prior_detector import ScribbleEdgeDetector
        detector = ScribbleEdgeDetector()
        super().__init__(detector=detector, **vpg_params)


class NullVPG(VisPriorGenerator):
    def __init__(self, vpg_params, detector_params=None):
        detector = lambda img:img
        super().__init__(detector=detector, **vpg_params)


class UniformerVPG(VisPriorGenerator):
    def __init__(self, vpg_params, detector_params=None):
        from .vis_prior_detector import UniformerMaskDetector
        detector = UniformerMaskDetector()
        super().__init__(detector=detector, **vpg_params)
        

class InpaintingVPG(VisPriorGenerator):
    def __init__(self, annotation, im_folder, device="cuda", ckpt_name="runwayml/stable-diffusion-inpainting"):
        from diffusers import StableDiffusionInpaintPipeline
        
        self.prior_bank = defaultdict(list)
        self.annotation = annotation
        self.im_folder = im_folder

        if ckpt_name is None:
            ckpt_name = "runwayml/stable-diffusion-inpainting"

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            ckpt_name, torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(device)

        self.update_prior_bank_with_images(annotation=annotation, im_folder=im_folder)

    def detect_one_img(self, img):
        return img
        
    def sample_images_to_inpaint(self, num_images, pixel_size):
        # reuse sample_image_visual_priors
        samples = self.sample_image_visual_priors(num_layouts=num_images, pixel_size=pixel_size)

        
        images = []
        bboxes = []
        categories = []
        prompts = []
        mask_images = []
        for i in range(num_images):
            sample = samples[i]
            if len(sample["bboxes"]) == 0:
                images.append(sample["vis_priors"][0])
                bboxes.append([])
                categories.append([])
                prompts.append("")
                mask_images.append(np.zeros_like(sample["vis_priors"][0]))
                continue
            
            image = sample["vis_priors"][0]
            bbox_idx = random.choice(range(len(sample["bboxes"])))  # TODO: try more than 1 
            bbox = sample["bboxes"][bbox_idx]
            cat = sample["cats"][bbox_idx]
            prompt = sample["cats"][bbox_idx]

            mask_image = self.sample_an_image_mask(image, [bbox])

            images.append(image)
            bboxes.append([bbox])
            categories.append([cat])
            prompts.append(prompt)
            mask_images.append(mask_image)

        # bboxes and cats here are for inpainting, the gt info of the image is in samples
        return images, mask_images, bboxes, categories, prompts, samples
    
    def sample_an_image_mask(self, image, bboxes):
        return get_bboxes_mask(img_shape = image.shape, bboxes = bboxes, value_in_bboxes=255, value_out_bboxes=0).astype(np.uint8)

    def inpaint_one_image(self, 
                            prompt, 
                            image, 
                            mask_image,
                            height=None,
                            width=None,
                            strength=1.,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            negative_prompt=None,
                            num_images_per_prompt=1,
                            eta=0.,
                           ):
        image = copy_numpy_to_pil(image)
        mask_image = copy_numpy_to_pil(mask_image)
        inpainted_images = self.pipe(prompt=prompt, 
                          image=image, 
                          mask_image=mask_image,
                          height=height,
                          width=width,
                          strength=strength,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          negative_prompt=negative_prompt,
                          num_images_per_prompt=num_images_per_prompt,
                          eta=eta,).images
        inpainted_images = [copy_pil_to_numpy(im) for im in inpainted_images]
        return inpainted_images
