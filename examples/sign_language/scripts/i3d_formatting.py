"""
This file follows Alvaro's approach to extracting sentence-level I3D from 
/mnt/gpid08/datasets/How2Sign/data_how2train/features/How2Sign/pretrained_bobsl/featurize-How2Sign_c1887_m_d3_prebobsl-v0-stride0.0625/filtered/1b3UpS1gD2k-8-rgb_front/
But saving them in the format that we need for the new dataloading (1 file per sentence) 
-> saved as .npy 1qMpH_7FL68_1-8-rgb_front.npy in the train/val/test folders
"""
#previous script in /home/usuaris/imatge/ltarres/wicv2023/how2sign/i3d_features/generate_npy_from_h5.py

import scipy.io
import json
import torch
import os
import sys
import cv2
import numpy as np

sentence_path = {
            'train': '/home/usuaris/imatge/ltarres/02_EgoSign/elan_files/SHARED_Laia/how2sign_realigned_train.csv',
            'val': '/home/usuaris/imatge/ltarres/02_EgoSign/elan_files/SHARED_Laia/how2sign_realigned_val.csv',
            'test': '/home/usuaris/imatge/ltarres/02_EgoSign/elan_files/SHARED_Laia/how2sign_realigned_test.csv',
}

path_to_i3d_save = {'train': '/home/usuaris/imatge/ltarres/wicv2023/how2sign/i3d_features/new/train/', 
               'val': '/home/usuaris/imatge/ltarres/wicv2023/how2sign/i3d_features/new/val/', 
               'test': '/home/usuaris/imatge/ltarres/wicv2023/how2sign/i3d_features/new/test/'}

#path_to_i3d_original = '/mnt/gpid08/datasets/How2Sign/data_how2train/features/How2Sign/' \
#           'pretrained_bobsl/featurize-How2Sign_c1887_m_d3_prebobsl-v0-stride0.0625/filtered/'

path_to_video = {'train': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/train/rgb_front/raw_videos/',
                'test': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/test/rgb_front/raw_videos/',
                'val': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/val/rgb_front/raw_videos/'
                }

def check_framerate(path_to_video_id, video_id):
    cap = cv2.VideoCapture(path_to_video_id + video_id + '.mp4')
    framerate = cap.get(cv2.CAP_PROP_FPS)
    return framerate

def extract_i3d_features(path_to_video_id, line):
    # path = '/mnt/gpid08/datasets/How2Sign/data_how2train/features/How2Sign/' \
    #        'pretrained_bsl1k/featurize-How2Sign-wlasl-msasl-c1079_m5_pret5K-v0-stride0.0625/filtered'
    path = '/mnt/gpid08/datasets/How2Sign/data_how2train/features/How2Sign/' \
           'pretrained_bobsl/featurize-How2Sign_c1887_m_d3_prebobsl-v0-stride0.0625/filtered/'
    
    #I should read this from the csv original
    video_name = line['VIDEO_NAME'] 
    fps = check_framerate(path_to_video_id, video_name) 
    start = max(0, int(line['START_REALIGNED']*fps) - 8)
    end = max(0, int(line['END_REALIGNED']*fps) - 8)
    print(line['SENTENCE_NAME'], fps, start, end, flush=True)

    name = os.path.join(path, video_name, 'features.mat')

    feat = scipy.io.loadmat(name)

    return feat['preds'][start:end+1]

def save_data(file_name, data):
    with open(file_name, 'wb') as f:
        np.save(f, data)
        
def save_i3d_for_videonames(path_to_i3d_save_i, path_to_video_id, videonames, data):
    samples = []
    for i, video_name in enumerate(videonames):
        print(f'Extracting video: {video_name} \n', flush=True)
        
        #Find the lines that correspond to the video_name
        lines = data[data['VIDEO_NAME'] == video_name]
        for index, line in lines.iterrows():
            sentence_name = line['SENTENCE_NAME']
            
            features = extract_i3d_features(path_to_video_id, line)
            
            if features.shape[0] > 0:
                #save features to npy
                name = os.path.join(path_to_i3d_save_i, sentence_name +'.npy')
                save_data(name, features)
            else:
                f = open(f'error_features_.txt', 'a')
                f.write(f'Error in line: {sentence_name} \n')
                f.close()
        print(f'--> Done! Finished currently: {i+1}/{len(videonames)} \n', flush=True)
    return samples

def load_original_csv(path):
    import pandas as pd
    df = pd.read_csv(path, sep='\t')
    return df

def load_i3d_video_names(path):
    with open(path, "r") as f:
        data_features = json.load(f)
    return data_features
        
def main(args):
    #Here, args[0] is the partition (train, val, test)
    dataframe = load_original_csv(sentence_path[args[0]])
    data_features = load_i3d_video_names('/home/usuaris/imatge/ltarres/sign2vec/fairseq-internal/examples/sign_language/scripts/subset2episode.json')
    #data features has all the names
    save_i3d_for_videonames(path_to_i3d_save[args[0]], path_to_video[args[0]], data_features[args[0]], dataframe)
    pass

if __name__ == "__main__":
    main(sys.argv[1:])