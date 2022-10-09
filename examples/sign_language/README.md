# Sign Language Fairseq (SL-Fairseq)

## First steps

Clone this repository, create the conda environment and install Fairseq:
```bash
git clone -b sign-language git@github.com:mt-upc/fairseq.git
cd fairseq

conda env create -f ./examples/sign_language/environment.yml
conda activate sign-language

pip install --editable .
```

The execution of scripts is managed with [Task](https://taskfile.dev/). Please follow the [installation instructions](https://taskfile.dev/installation/) in the official documentation.

We recommend using sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b path-to-env/sign-language-new/bin

## 🏗 Work in Progress

## Citations
- Some scripts from this repository use the GNU Parallel software.
  > Tange, Ole. (2022). GNU Parallel 20220722 ('Roe vs Wade'). Zenodo. https://doi.org/10.5281/zenodo.6891516

## To extract the poses
  
  ```python
  cd sign2vec/fairseq-internal/examples/sign_language
  srun -p gpi.develop --time 01:00:00 --mem 100G --gres=gpu:1 -c 8 --pty bash #Remember to use the compute node if you need to run for longer
  conda activate sign-language-new
  export EGOSIGN_DIR=/mnt/gpid08/datasets/How2Sign/EgoSign
  task egosign:val:extract_mediapipe
  ```
  If this command gives you error (/bin/bash: line 1: 27052 Killed) it is probably because you are not providing enough disk space --mem. Try incrementing the space or reducing the number of parallel jobs.


#Solving the error of extract_mediapipe
I will make an experiment, changing their library, how they extract mediapipe. To do it, I need to use their github, not directly import pose_format

from pose_format.utils.holistic import load_holistic %this is what we need to change. For the following:
import sys
sys.path.insert(1, '../../../../EgoSign/visualization/poseformat/')
from pose_format.utils.holistic import load_holistic

Let's try this again:
python ./scripts/extract_mediapipe.py --video-file /mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/val/2SnVWW3MOB4-12/2SnVWW3MOB4-12-2022-06-20_13-14-38-rgb_front.mp4 --poses-file ../../../../EgoSign/visualization/poseformat/2SnVWW3MOB4-rgb_front_laia.pose --fps 30

Triga 4 minuts en processar-se el vídeo.
Després fem visualize_mediapipe.ipynb per generar els mp4 i visualitzar-los.

#Anem a fer les visualitzacions, per què no es veu bé?
Provem amb una versió reduïda del mateix video:
conda install -c conda-forge imageio
conda install -c conda-forge imageio-ffmpeg
python ./scripts/extract_mediapipe.py --video-file ../../../../EgoSign/visualization/examples_laia/2SnVWW3MOB4-12-rgb_front_cut.mp4 --poses-file ../../../../EgoSign/visualization/examples_laia/2SnVWW3MOB4-12-rgb_front_cut.pose --fps 30


#Anem a provar-ho tot amb les diferents views:
cd sign2vec/fairseq-internal/examples/sign_language
srun -p gpi.develop --time 01:00:00 --mem 100G --gres=gpu:1 -c 8 --pty bash
conda activate sign-language-new

#EgoSign front
python ./scripts/extract_mediapipe.py --video-file /mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/val/2SnVWW3MOB4-12/2SnVWW3MOB4-12-2022-06-20_13-14-38-rgb_front.mp4 --poses-file ../../../../EgoSign/visualization/examples_laia/2SnVWW3MOB4-rgb_front_egosign.pose --fps 30

#EgoSign side
python ./scripts/extract_mediapipe.py --video-file /mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/val/2SnVWW3MOB4-12/2SnVWW3MOB4-12-2022-06-20_13-14-38-rgb_side.mp4 --poses-file ../../../../EgoSign/visualization/examples_laia/2SnVWW3MOB4-rgb_side_egosign.pose --fps 30

#EgoSign head
python ./scripts/extract_mediapipe.py --video-file /mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/val/2SnVWW3MOB4-12/2SnVWW3MOB4-12-2022-06-20_13-14-38-rgb_head.mp4 --poses-file ../../../../EgoSign/visualization/examples_laia/2SnVWW3MOB4-rgb_head_egosign.pose --fps 30

#How2Sign front
python ./scripts/extract_mediapipe.py --video-file /mnt/gpid08/datasets/How2Sign/How2Sign/video_level/val/rgb_front/raw_videos/2SnVWW3MOB4-2-rgb_front.mp4 --poses-file ../../../../EgoSign/visualization/examples_laia/2SnVWW3MOB4-rgb_front_how2sign.pose --fps 25

#Anem a veure que les poses estigui agafant dreta/esquerre i keypoint que toca
