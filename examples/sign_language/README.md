# Sign Language Translation from Instructional Videos

This repository contains the implementation for the Sign Language Translation from Instructional Videos paper. The citation of the paper is at the end of this README.
All the scripts are located inside examples/sign_language/scripts.

## First steps
Clone this repository, create the conda environment and install Fairseq:
```bash
git clone -b slt_how2sign_wicv2023 git@github.com:mt-upc/fairseq.git
cd fairseq

conda env create -f ./examples/sign_language/environment.yml
conda activate slt-how2sign-wicv2023

pip install --editable .
```

The execution of scripts is managed with [Task](https://taskfile.dev/). Please follow the [installation instructions](https://taskfile.dev/installation/) in the official documentation.
We recommend using the following
```bash
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b path-to-env/sign-language-new/bin
```

## Downloading the data
We are working on uploading the I3D keypoints and .tsv to the dataverse. Once you have them, they should follow this structure:
```
├── data/
│   ├── train/
│   │   ├── --7E2sU6zP4_10-5-rgb_front.npy
│   │   ├── --7E2sU6zP4_11-5-rgb_front.npy
│   │   └── ...
│   ├── val/
│   │   ├── -d5dN54tH2E_0-1-rgb_front.npy
│   │   ├── -d5dN54tH2E_1-1-rgb_front.npy
│   │   └── ...
│   └── test/
│       ├── -fZc293MpJk_0-1-rgb_front.npy
│       ├── -fZc293MpJk_1-1-rgb_front.npy
│       └── ...
├── cvpr23.fairseq.i3d.train.how2sign.tsv
├── cvpr23.fairseq.i3d.val.how2sign.tsv
└── cvpr23.fairseq.i3d.test.how2sign.tsv
```

Each of the folder partitions contain the corresponding I3D features in .npy files, provided by [previous work](https://imatge-upc.github.io/sl_retrieval/), that correspond to each How2Sign sentence.  
In addition, we provide the `.tsv` files for all the partitions that contains the metadata about each of the sentences, such as translations, path to `.npy` file, duration. 
Notice that you might need to manually change the path of the `signs_file` column.

## Training the corresponding sentencepiece model
Given that our model operated on preprocessed text, we need to build a tokenizer with a lowercased text.
```bash
cd FAIRSEQ_ROOT/experiments/sign_language/
task how2sign:train_sentencepiece_lowercased
```
Previously to the call of the function, a `FAIRSEQ_ROOT/examples/sign_language/.env` file should be defined with the following variables:
```bash
FAIRSEQ_ROOT: path/to/fairseq
SAVE_DIR: path/to/tsv
VOCAB_SIZE: 7000
```
As you have read in the paper, we are using rBLEU as a metric. The blacklist can be found in: `FAIRSEQ_ROOT/examples/sign_language/scripts/blacklisted_words.txt`

## Training 
As per fairseq documentation, we work with config files that can be found in `CONFIG_DIR = FAIRSEQ_ROOT/examples/sign_language/config/i3d_best`. Select the name of the .yaml files as the experiment name desired. Simply select an EXPERIMENT_NAME and run:
```bash
export EXPERIMENT=${EXPERIMENT_NAME}
task train_slt
```
Remember to have a GPU available and the environment activated.
Previously to the call of the function, the .env should be updated with the following variables:
```bash
DATA_DIR: path/to/i3d/folders
WANDB_ENTITY: name/team/WANDB
WANDB_PROJECT: name_project_WANDB
NUM_GPUS: 1
CONFIG_DIR: FAIRSEQ_ROOT/examples/sign_language/config/i3d_best
```

## Evaluation
```bash
export EXPERIMENT=${EXPERIMENT_NAME}
task generate
```
Similarly to other tasks, the .env should be updated:
```bash
EXPERIMENT: 
CKPT:
SUBSET:
SPM_MODEL:
```

## Citations
- If you fins this work useful, please consider citing:
<i>
Laia Tarres, Gerard I. Gallego, Amanda Duarte, Jordi Torres and Xavier Giro-i-Nieto. "Sign Language Translation from Instructional Videos", WCVPR 2023.
</i>
<pre>
@InProceedings{slt-how2sign-wicv2023,
author = {Tarres, Laia and Gallego, Gerard and Duarte, Amanda and Torres, Jordi and Giro-i-Nieto, Xavier},
title = {Sign Language Translation from Instructional Videos},
booktitle = {Workshops on the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2023}
}
</pre>
- Some scripts from this repository use the GNU Parallel software.
  > Tange, Ole. (2022). GNU Parallel 20220722 ('Roe vs Wade'). Zenodo. https://doi.org/10.5281/zenodo.6891516

Check the original [Fairseq README](https://github.com/mt-upc/fairseq/blob/wicv2023-slt/README_FAIRSEQ.md) to learn how to use this toolkit.