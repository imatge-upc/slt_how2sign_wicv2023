# Sign2vec

## Set up the environment

Clone this repository to your machine:
```bash
export FAIRSEQ_ROOT=... # Set this to the directory where you want to clone fairseq
export S2V_DIR=${FAIRSEQ_ROOT}/examples/sign2vec

git clone -b sign2vec git@github.com:mt-upc/fairseq-internal.git ${FAIRSEQ_ROOT}
```

Create the environment and activate it:
```bash
conda env create -f ${S2V_DIR}/environment.yml && \
conda activate sign2vec
```

Install fairseq:
```bash
pip install --editable ${FAIRSEQ_ROOT}
```

Define the root folder of [How2Sign](https://how2sign.github.io):
```bash
export H2S_ROOT=...
```

The data in `H2S_ROOT` should be organized as follows:
```bash
${H2S_ROOT}
├── test.tsv
├── test.h5
├── train.tsv
├── train.h5
├── val.tsv
└── val.h5
```

Where `h5` files contain the keypoints at video level and `tsv` files contain the text translations and information about the start and end of sentences.


## Prepare the data

Execute the following script to perform the following actions to the data:
- Split the video level keypoints into sentence level keypoints
- Filter out to short or too long examples
- Generate the vocabulary

```bash
python ${FAIRSEQ_ROOT}/examples/sign2vec/prep_how2sign.py \
    --data-root ${H2S_ROOT} \
    --min-n-frames 5 \
    --max-n-frames 4000 \
    --vocab-type unigram \
    --vocab-size 4000 \
```

After the script finishes, the data in `H2S_ROOT` should be organized as follows:

```bash
${H2S_ROOT}
├── test.tsv
├── test_filt.tsv
├── test_sent.h5
├── test.h5
├── train.tsv
├── train_filt.tsv
├── train_sent.h5
├── train.h5
├── val.tsv
├── val_filt.tsv
├── val_sent.h5
└── val.h5
```

Where files ending with `_sent.h5` are the keypoints at sentence level, arrays of shape (seq_len, num_keypoints*4), and `_filt.tsv` are the filtered tsv files.

## [Work in Progress]
