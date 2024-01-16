import cv2 as cv
import numpy as np
import random
from PIL import Image

MINIMAL_WIDTH = 5  # in original image
MINIMAL_HEIGHT = 5  # in original image
MINIMAL_AREA_RATIO = 0.01 * 0.01  # in resized image

def imread(im_path):
    return cv.imread(im_path)[:,:,::-1]


def imwrite(im_path, img):
    return cv.imwrite(im_path, img[:,:,::-1])


def resize(img, new_W, new_H, anno=None):
    new_img = cv.resize(img, (new_W, new_H))
    if new_img.ndim == 2:  # cv2 resize will kill empty dimension of visual prior
        new_img = new_img[:, :, None]
    return new_img


def copy_cvbgr_to_pil(img):
    im_np = img.copy()
    im_np = cv.cvtColor(im_np, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def copy_numpy_to_pil(img):
    im_np = img.copy()
    im_pil = Image.fromarray(img)
    return im_pil


def copy_pil_to_cv(img):
    im_pil = img.copy()
    im_np = np.asarray(im_pil)[:,:,::-1]
    return im_np


def copy_pil_to_numpy(img):
    im_pil = img.copy()
    im_np = np.asarray(im_pil)
    return im_np


def get_bboxes_mask(img_shape, bboxes, value_in_bboxes, value_out_bboxes):
    mask = np.zeros(img_shape)
    mask += value_out_bboxes
    for bbox in bboxes:
        x, y, w, h = bbox
        mask[int(y):int(y+h), int(x):int(x+w), :] = value_in_bboxes
    return mask


def crop_bboxes(img, bboxes):
    # bboxes [[x1,y1,w,h],[x1,y1,w,h],...]
    # only keep bboxes areas of the image

    mask = get_bboxes_mask(img_shape=img.shape, bboxes=bboxes, value_in_bboxes=1, value_out_bboxes=0)

    return img * mask


def set_seed(seed):
    # seed everything
    import random
    import numpy as np
    import os
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_crop_and_resize_images_to_square(imgs, square_size, bboxes=None, cats=None):
    if bboxes is not None:
        assert len(imgs) == 1  # for now only support one image input if bbox is provided
        assert cats is not None
        assert len(cats) == len(bboxes)

    H, W, _ = imgs[0].shape
    length = min(W, H)

    img_results = []
    bboxes_results = []
    cats_results = []

    for img_origin in imgs:
        img = img_origin.copy()

        h0 = int(random.uniform(0, H - length))
        w0 = int(random.uniform(0, W - length))

        img = img[h0:h0+length, w0:w0+length, :]
        img = resize(img=img, new_W=square_size, new_H=square_size)

        if bboxes is not None:
            for i in range(len(bboxes)):
                x, y, w, h = bboxes[i]
                cat_name = cats[i]

                # move bbox:
                y -= h0
                x -= w0
                # skip bbox if it is out of boundary of the square image
                if y + h < MINIMAL_HEIGHT or x + w < MINIMAL_WIDTH:
                    continue

                # crop out of boundary part of bbox 
                if y < 0:
                    h = h - (-y)
                    y = 0
                if x < 0:
                    w = w - (-x)
                    x = 0

                # resize
                scale = square_size/length
                x = int(x * scale)
                y = int(y * scale)
                w = int(w * scale)
                h = int(h * scale)

                # skip the bbox if area is too small
                if w * h < MINIMAL_AREA_RATIO * square_size * square_size:
                    continue

                bboxes_results.append([x, y, w, h])
                cats_results.append(cat_name)

        img_results.append(img)

    if bboxes is None:
        return img_results
    else:
        return img_results, bboxes_results, cats_results


def get_allinfo_of_an_image(coco, img_obj):

    annIds = coco.getAnnIds(imgIds=[img_obj["id"]])
    anns = coco.loadAnns(annIds)
    bboxes = [ann["bbox"] for ann in anns]

    catIds = [ann["category_id"] for ann in anns]
    cats = coco.loadCats(catIds)
    cat_names = [cat["name"] for cat in cats]

    ret = {}
    ret["bboxes"] = bboxes
    ret["anns"] = anns
    ret["annIds"] = annIds
    ret["catIds"] = catIds
    ret["cats"] = cats
    ret["cat_names"] = cat_names

    return ret


def nms(x, t, s):
    x = cv.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z
