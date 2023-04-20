#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from examples.speech_to_text.data_utils import gen_vocab

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-file", required=True, nargs='+', type=str)
    parser.add_argument("--spm-prefix", required=True, type=str)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--vocab-type", default="unigram", type=str,
                        choices=["bpe", "unigram", "char", "word"])
    parser.add_argument("--column", default="translation", type=str)
    parser.add_argument("--lowercase", default=False, type=bool)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    sentences = []
    for tsv_file in args.tsv_file:
        tsv_file = Path(tsv_file).expanduser().resolve()
        df = pd.read_csv(tsv_file, sep='\t')
        sentences.extend(df[args.column].to_list())

    with NamedTemporaryFile(mode="w") as f:
        for sent in sentences:
            if args.lowercase:
                sent = sent.lower()
            f.write(sent + "\n")

        gen_vocab(
            Path(f.name),
            Path(args.spm_prefix),
            args.vocab_type,
            args.vocab_size,
        )


if __name__ == "__main__":
    main()
