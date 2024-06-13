# [project name]
based on the work of [3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting](https://arxiv.org/abs/2312.09228).
## [Original Paper](https://arxiv.org/abs/2312.09228) | [Project Page](https://neuralbodies.github.io/3DGS-Avatar/index.html)


## Installation
### Environment Setup
This repository has been tested on the following platform:
1) Python 3.7.13, PyTorch 1.12.1 with CUDA 11.6 and cuDNN 8.3.2, Ubuntu 22.04/CentOS 7.9.2009

To clone the repo, run either:
```
git clone --recursive https://github.com/tanshiv2/smplx_human_avatar.git
```

Next, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `3dgs-avatar` using
```
conda env create -f environment.yml
conda activate 3dgs-avatar
# install tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### SMPL Setup
Download `SMPL v1.0 for Python 2.7` from [SMPL website](https://smpl.is.tue.mpg.de/) (for male and female models), and `SMPLIFY_CODE_V2.ZIP` from [SMPLify website](https://smplify.is.tue.mpg.de/) (for the neutral model). After downloading, inside `SMPL_python_v.1.0.0.zip`, male and female models are `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` and `smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl`, respectively. Inside `mpips_smplify_public_v2.zip`, the neutral model is `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`. Remove the chumpy objects in these .pkl models using [this code](https://github.com/vchoutas/smplx/tree/master/tools) under a Python 2 environment (you can create such an environment with conda). Finally, rename the newly generated .pkl files and copy them to subdirectories under `./body_models/smpl/`. Eventually, the `./body_models` folder should have the following structure:
```
body_models
 └-- smpl
    ├-- male
    |   └-- model.pkl
    ├-- female
    |   └-- model.pkl
    └-- neutral
        └-- model.pkl
```

Then, run the following script to extract necessary SMPL parameters used in our code:
```
python extract_smpl_parameters.py
```
The extracted SMPL parameters will be saved into `./body_models/misc/`.

## Dataset preparation

### ZJU-MoCap
Due to license issues, we cannot publicly distribute our preprocessed ZJU-MoCap and PeopleSnapshot data. 
Please follow the instructions of [ARAH](https://github.com/taconite/arah-release) to download and preprocess the datasets.
For PeopleSnapshot, we use the optimized SMPL parameters from Anim-NeRF [here](https://drive.google.com/drive/folders/1tbBJYstNfFaIpG-WBT6BnOOErqYUjn6V?usp=drive_link).

### X Humans
For the use of X-Humans dataset, please refer to the official [website](https://xhumans.ait.ethz.ch) for terms of use and for download application.

After downloading the dataset, run the following command in the project root directory:
```
export PYTHONPATH=${PWD}
python ./preprocess_datasets/preprocess_xhumans.py
```
In `./preprocess_datasets/preprocess_xhumans.py`, you can change the data path, the split (train/test) to be processed, and the human body model type (smpl/smplx).

Additionally, hand masks need to be generated to calculate hand-aware losses. Run the following in the project root directory:
```
python render_mesh.py
```
You can also change the data path and the split in this file.

## Training Configurations

### dataset
can be specified with command line option dataset=data_config_filename
the corresponding configuration file data_config_filename.yaml should be created under the configs/dataset/ folder, which specifies the target directory, model type, and traning/validation input division.

### models
for training you need to specify the models of the deformer and pose correction modules.
```
rigid={identity: identity mapping for single frame; smpl_nn: use KNN for weights prediction; skinning_field_smplx: use a network for weights prediction}
non_rigid={identity: no deformation; mlp: a single layer mlp; hashgrid: hashgrid encoding, for SMPL; hanshgrid_smplx: for SMPLX}
pose_correction={none: pose correction disabled, direct: pose correction enabled, for SMPL; direct_smplx: pose correction enabled, for SMPLX}
```

the yaml files for each are inside the corresponding folder under configs directory. Parameters and input dimension can be adjusted by the user, via creating new files following similar manners.

### training options
The files under the folder configs/option specifies training parameters like training iterations, operation delay, learning rate, test intervals.
can be specified with option=option_filename

**example usage**
```shell
# ZJU-MoCap
python train.py dataset=zjumocap_377_mono
# PeopleSnapshot
python train.py dataset=ps_female_3 option=iter30k pose_correction=none
# X Humans
python train.py dataset=x_humans_00036_1 non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none option=iter40k
```
To train on a different subject, simply choose from the configs in `configs/dataset/`.

### wandb setup
We use [wandb](https://wandb.ai) for online logging, which is free of charge but needs online registration.
In the train.py script, you can edit the login setting in the main function.

## Evaluation
To evaluate the method for a specified subject, run
```shell
# ZJU-MoCap
python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono
# PeopleSnapshot
python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ps_female_3
# X Humans
python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=x_humans_00036_8
```

## Test on out-of-distribution poses
First, please download the preprocessed AIST++ and AMASS sequence for subjects in ZJU-MoCap [here](https://drive.google.com/drive/folders/17vGpq6XGa7YYQKU4O1pI4jCMbcEXJjOI?usp=drive_link) 
and extract under the corresponding subject folder `${ZJU_ROOT}/CoreView_${SUBJECT}`.

To animate the subject under out-of-distribution poses, run
```shell
python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

We provide four preprocessed sequences for each subject of ZJU-MoCap, 
which can be specified by setting `dataset.predict_seq` to 0,1,2,3, 
where `dataset.predict_seq=3` corresponds to the canonical rendering.

Currently, the code only supports animating ZJU-MoCap models for out-of-distribution models.

## License
We employ [MIT License](LICENSE) for the 3DGS-Avatar code, which covers
```
configs
dataset
models
utils/dataset_utils.py
extract_smpl_parameters.py
render.py
train.py
```

The rest of the code are modified from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 
Please consult their license and cite them.

## Acknowledgement
This project is built on source codes from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 
We also use the data preprocessing script and part of the network implementations from [ARAH](https://github.com/taconite/arah-release).
We sincerely thank these authors for their awesome work.

