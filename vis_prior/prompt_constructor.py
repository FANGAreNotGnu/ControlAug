import random
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

class PromptConstructor:
    def __init__(self, valid_prompt_modes=None, prompt_mode=None, device="cuda:0") -> None:
        if valid_prompt_modes is not None:
            self.VALID_PRMOPT_MODES = valid_prompt_modes  # this is important to control what to mix
        else:
            self.VALID_PRMOPT_MODES = ["cat", "and", "set", "setand", "shuffledset", "shuffledsetand", "img", "pic", "mix"]
        self.prompt_mode = prompt_mode
        self.device = device
        self.blip_processor = None
        self.blip_model = None

    def generate_blip_prompts(self, img, prompt_mode=None):
        if prompt_mode is None:
            prompt_mode = self.prompt_mode
        if prompt_mode == "blip_large":
            model_name = "Salesforce/blip-image-captioning-large"
        elif prompt_mode == "blip_base":
            model_name = "Salesforce/blip-image-captioning-base"
        else:
            raise ValueError("only support checkpoints from blip")

        if self.blip_processor is None:
            self.blip_processor = AutoProcessor.from_pretrained(model_name)
        if self.blip_model is None:
            self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

        inputs = self.blip_processor(images=img, text="a picture of", return_tensors="pt").to(self.device)
        outputs = self.blip_model.generate(**inputs)
        prompt = self.blip_processor.decode(outputs[0], skip_special_tokens=True).strip()

        return prompt

    def __call__(self, cats, img, prompt_mode=None):
        '''
        cats: a list of category names
        '''

        if prompt_mode is None:
            prompt_mode = self.prompt_mode

        if "blip" in prompt_mode:
            return self.generate_blip_prompts(img=img, prompt_mode=prompt_mode)

        assert prompt_mode in self.VALID_PRMOPT_MODES

        if prompt_mode == "cat":    
            return [", ".join(cats)] 
        elif prompt_mode == "and":    
            return [" and ".join(cats)] 
        elif prompt_mode == "set":    
            return [", ".join(list(set(cats)))] 
        elif prompt_mode == "setand":    
            return [" and ".join(list(set(cats)))] 
        elif prompt_mode == "shuffledset":
            cat_nodup = list(set(cats))
            random.shuffle(cat_nodup)
            return [", ".join(cat_nodup)] 
        elif prompt_mode == "shuffledsetand":
            cat_nodup = list(set(cats))
            random.shuffle(cat_nodup)
            return [" and ".join(cat_nodup)] 
        elif prompt_mode == "img":
            cat_nodup = list(set(cats))
            random.shuffle(cat_nodup)
            return ["An image of " + ", ".join(cat_nodup)]
        elif prompt_mode == "pic":
            cat_nodup = list(set(cats))
            random.shuffle(cat_nodup)
            return ["A picture of " + ", ".join(cat_nodup)]
        elif prompt_mode == "mix":
            return self.generate_prompts_from_cats(cats=cats, prompt_mode=random.choice(self.VALID_PRMOPT_MODES[:-1]))
        else:
            raise ValueError(f"d{prompt_mode} is not supported.")
