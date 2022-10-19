#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path

import pandas as pd

from examples.sign_language.datasets import (
    get_dataset,
    SignLanguageDataset,
    DATASET_REGISTRY
)
from examples.speech_to_text.data_utils import save_df_to_tsv

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASET_REGISTRY.keys(), type=str)
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--tsv-out", required=True, type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--split", required=True, nargs='+', type=str)
    parser.add_argument("--signs-lang", type=str)
    parser.add_argument("--translation-lang", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    base_dir = Path(args.path).expanduser().resolve()
    tsv_out = Path(args.tsv_out).expanduser().resolve()

    df_out = pd.DataFrame(columns=SignLanguageDataset.MANIFEST_COLUMNS)
    dataset_cls = get_dataset(args.dataset) #aquí hi ha: __module__, __doc__, LANGUAGES, SPLITS, SIGNS_TYPES, SIGNS_LANGS, TRANSLATION_LANGS, __init__

    #for s in args.split:
    s = args.split[0]
    log.info(f"Processing '{s}' split from '{args.dataset}' dataset")
    dataset = dataset_cls(base_dir, s, args.type, args.signs_lang, args.translation_lang)
    df_out = pd.concat((df_out, dataset.data), ignore_index=True)

    tsv_out.parent.mkdir(parents=True, exist_ok=True)
    save_df_to_tsv(df_out, tsv_out)
    log.info(f"Generated TSV file saved to: {tsv_out}")

if __name__ == "__main__":
    main()
