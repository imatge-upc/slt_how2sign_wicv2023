#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2

#from pose_format.utils.holistic import load_holistic #We want to use our own implementation
import sys
sys.path.insert(1, '/home/usuaris/imatge/ltarres/02_EgoSign/visualization/poseformat/') #This path will need to change if we try to run it in Amada's user
from pose_format.utils.holistic import load_holistic


def extract_poses(vid_file: Path, new_fps: int):
    video = cv2.VideoCapture(vid_file.as_posix())
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Video FPS: {fps}")

    success, image = video.read()
    frames = []
    while success:
        frames.append(image)
        success, image = video.read()

    poses = load_holistic(
        frames,
        fps=fps,
        width=frames[0].shape[1],
        height=frames[0].shape[0],
        depth=10,
        progress=True,
        additional_holistic_config={
            'min_detection_confidence': 0.2,
            'min_tracking_confidence': 0.3,
        }
    )
    #poses = poses.interpolate(new_fps, kind="quadratic") #I don't think for the egosign this is necessary
    return poses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-file', type=str, required=True, help="Input video file (MP4)")
    parser.add_argument('--poses-file', type=str, required=True, help="File where the extracted poses will be saved")
    parser.add_argument('--fps', type=int, default=24, help="Target FPS for poses")
    return parser.parse_args()


def main():

    args = parse_args()

    video_file = Path(args.video_file).expanduser().resolve()
    poses_file = Path(args.poses_file).expanduser().resolve()

    assert video_file.is_file(), "The input file does not exist"
    poses_file.parent.mkdir(parents=True, exist_ok=True)
    poses = extract_poses(video_file, args.fps)
    with open(poses_file.as_posix(), "wb") as f:
        poses.write(f)


if __name__ == '__main__':
    main()
