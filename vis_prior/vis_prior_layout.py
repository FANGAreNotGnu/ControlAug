import random


class VisPriorLayout():
    pass


# NoClip: bbox not going out of image bound, thus no clip of bbox needed
class UniformRandomNoClipVPL(VisPriorLayout):
    def __init__(self, scale_down = 1, scale_up = 1, min_ratio = -1):
        '''
        scale_down, scale_up: new_object/old_object
        min_ratio: min(object/image)
        '''
        self.scale_down = scale_down
        self.scale_up = scale_up
        self.min_ratio = min_ratio

    def generate_a_layout_with_prior(self, im_shape, priors, num_object):
        layout = []
        object_categories = random.choices(list(priors.keys()), k=num_object)
        for category_name in object_categories:
            im_prior = random.choice(priors[category_name])
            bbox = self.generate_bbox(im_shape=im_shape, prior_shape=im_prior.shape[:2])
            layout.append([category_name, bbox, im_prior])
        
        return layout

    def generate_layouts_with_prior(self, im_shape, priors, num_object, num_layouts):
        layouts = []
        for i in range(num_layouts):
            layout = self.generate_a_layout_with_prior(im_shape, priors, num_object)
            layouts.append(layout)

        return layouts

    def generate_bbox(self, im_shape, prior_shape):

        h, w = prior_shape
        H, W, _ = im_shape

        assert W >= 0
        assert H >= 0
        assert w >= 0
        assert h >= 0

        w_max = min(w * self.scale_up, W-1)
        h_max = min(h * self.scale_up, H-1)
        w_min = min(w * self.scale_down, W-1)
        h_min = min(h * self.scale_down, H-1)

        max_scale = min(self.scale_up, (W-1)/w, (H-1)/h)
        if self.min_ratio > 0:
            assert self.min_ratio < 1
            min_scale = max(self.scale_down, self.min_ratio * (W-1)/w, self.min_ratio * (H-1)/h)  # w * min_scale >= self.min_ratio * (W-1)
            min_scale = min(min_scale, (W-1)/w, (H-1)/h)
        else:
            min_scale = min(self.scale_down, (W-1)/w, (H-1)/h)
        scale = random.uniform(min_scale, max_scale)

        w_new = int(scale * w)
        h_new = int(scale * h)

        x_max = W - w_new - 1
        y_max = H - h_new - 1    

        x_new = int(random.random() * x_max)
        y_new = int(random.random() * y_max)

        return x_new, y_new, w_new, h_new,

