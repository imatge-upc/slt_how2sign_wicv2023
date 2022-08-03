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

## 🏗 Work in Progress

## Citations
- Some scripts from this repository use the GNU Parallel software.
  > Tange, Ole. (2022). GNU Parallel 20220722 ('Roe vs Wade'). Zenodo. https://doi.org/10.5281/zenodo.6891516
