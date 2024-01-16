# ControlAug
A Pipeline of Data Augmentation with Diffusion Model for Few Shot Object Detection


## Preparations


### Setup

Clone current repository

`git clone https://github.com/FANGAreNotGnu/ControlAug.git`

Clone ControlNet repository. Make sure all repositories are under the same folder.

`git clone https://github.com/FANGAreNotGnu/ControlNet.git`

Clone MMDetection repository. Make sure all repositories are under the same folder.

`git clone https://github.com/FANGAreNotGnu/mmdetection.git`

Create a Conda Environment for ControlNet (environment name: ControlAug_control)

`conda env create -f ControlNet/environment.yaml`

Create a Conda Environment for CLIP (environment name: ControlAug_clip).
`conda env create -f ControlAug/environment/ControlAug_clip.yaml`

Create a Conda Environment for Diffuser (environment name: ControlAug_diffuser).
`conda env create -f ControlAug/environment/ControlAug_diffuser.yaml`

Create a Conda Environment for MMDetection (environment name: ControlAug_mmdet).

```
conda env create -f ControlAug/environment/ControlAug_mmdet.yaml
conda activate ControlAug_mmdet
mim install mmcv==2.0.1
pip install mmdet==3.1.0
conda deactivate ControlAug_mmdet
```

Export Paths

`source ./ControlAug/scripts/export_paths.sh`


### Download

#### Download Data

Download COCO FSOD. Make sure paths are exported.

`bash ./ControlAug/scripts/download_coco_fsod.sh`

#### Download ControlNet Checkpoints

Download ControlNet Checkpoints. Make sure paths are exported.

`bash ./ControlAug/scripts/download_cnet_ckpts.sh`



## Run Augmentation Pipeline

`bash ./coco10_cat_pipe.sh 0 1 10 512 333 HED blip_large_coco 30 control_sd15_hed.pth`
