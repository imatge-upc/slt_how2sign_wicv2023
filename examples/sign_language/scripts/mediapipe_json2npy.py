#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import multiprocessing
from pathlib import Path

import numpy as np

LMKS = {
    'face': 128,        # FIXME: This is just for the landmarks given by WMT-SLT (468 if extracted)
    'pose': 33,
    'left_hand': 21,
    'right_hand': 21,
}


def json_to_numpy(in_file):
    f = open(in_file.as_posix(), 'r')
    data = json.load(f)

    frame_landmarks = []
    for bp in LMKS.keys():
        bp_landmarks = [
            [float(n) for n in lm.split(',')[:3]]
            for lm in data[f'{bp}_landmarks']['landmarks']
        ]
        if len(bp_landmarks) == 0:
             # TODO: Decide if we want to do this
            bp_landmarks = [[-1.0, -1.0, -1.0]] * LMKS[bp]

        frame_landmarks += bp_landmarks

    return np.array(frame_landmarks)


def main(args):
    in_dir = Path(args.input).expanduser().resolve()
    out_file = Path(args.output).expanduser().resolve()

    json_files = list(sorted(in_dir.glob("**/*.json")))
    with multiprocessing.Pool() as p:
        npy_array = np.array(list(p.map(json_to_numpy, json_files)))

    with open(out_file.as_posix(), 'wb') as f:
        np.save(f, npy_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory containing mediapipe JSON files corresponding to a video')
    parser.add_argument('--output', type=str, help='Path to the generated NumPy output')
    args = parser.parse_args()

    main(args)
