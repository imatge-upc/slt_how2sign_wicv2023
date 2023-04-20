from argparse import Namespace
import argparse
import os
from sacremoses import MosesDetokenizer
import truecase
from fairseq.data import encoders
import sacrebleu
#from tqdm import tqdm
from fairseq.tasks.sign_to_text import SignToTextConfig, SignToTextTask

from sacrebleu.metrics import BLEU, CHRF, TER
#import pandas as pd

def decode(tokens, bpe_tokenizer, moses_detok):
    '''Decode the output tokens into a decoded string.'''
    if bpe_tokenizer:
        tokens = bpe_tokenizer.decode(tokens)
    if moses_detok:
        #tokens = moses_detok.detokenize(tokens.split())
        tokens = truecase.get_true_case(tokens)
    return tokens

def parse_generate_file(generate_file, backlist_file, partition, path_to_vocab):
    '''Returns all H and T lines found in the generate file, grouped by id'''
    
    config = SignToTextConfig()
    config.bpe_sentencepiece_model = path_to_vocab
    config.data = "/home/usuaris/imatge/ltarres/wicv2023/how2sign/i3d_features"
    task = SignToTextTask.setup_task(config)
    task.load_dataset(f"cvpr23.fairseq.i3d.{partition}.how2sign")
    bpe_tokenizer = encoders.build_bpe(
            Namespace(
                bpe='sentencepiece',
                sentencepiece_model=config.bpe_sentencepiece_model
            )
    )
    moses_detok = MosesDetokenizer(lang='en')

    with open(generate_file, 'r') as file:
        generate_lines = file.read().split("\n")
    with open(backlist_file, 'r') as file:
        blacklisted_words = file.read().split("\n")
    
    dict_generated = {}
    idx=0
    for line in generate_lines:
        #Check if we should skip the line
        if line.startswith('Generate'):
            break
        idx = line.split('\t')[0].split('-')[1]
        if idx not in dict_generated.keys():
            dict_generated[idx] = {}
        if line.startswith('H'):
            pred = line.split('\t')[2]
            pred_processed = decode(pred, bpe_tokenizer, moses_detok)
            no_pred = [word for word in pred_processed.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')).split(' ') if word not in blacklisted_words]
            if len(no_pred) == 0:#If the list is empty, we add a space to avoid errors
                no_pred.append(' ')
            if idx in dict_generated.keys():
                dict_generated[idx]['pred'] = pred_processed
                dict_generated[idx]['no_pred'] = " ".join(no_pred)
                
        elif line.startswith('T'): #We should check this from the dataset, as we want the raw text
            ref_lowercased = line.split('\t')[1]
            #check from the dataset, this id and see if it is the same but without lowercasing
            real_ref = task.datasets[f'cvpr23.fairseq.i3d.{partition}.how2sign'].get_label(int(idx))
            file_rgb = task.datasets[f'cvpr23.fairseq.i3d.{partition}.how2sign'][int(idx)]['vid_id']
            no_ref= [word for word in real_ref.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')).split(' ') if word not in blacklisted_words]
            if idx in dict_generated.keys():
                dict_generated[idx]['ref'] = real_ref
                dict_generated[idx]['no_ref'] = " ".join(no_ref)
                dict_generated[idx]['file_rgb'] = file_rgb
                
    return dict_generated

def compute_metrics(dict_generated):
    '''Computes the different metrics following the valid_step of our task'''
    preds, refs, preds_reduced, refs_reduced = [], [], [], []
    for idx in sorted(dict_generated.keys()):
        pred = dict_generated[idx]['pred']
        no_pred = dict_generated[idx]['no_pred']
        ref = dict_generated[idx]['ref']
        no_ref = dict_generated[idx]['no_ref']
        bleu = BLEU().corpus_score([pred], [[ref]])
        dict_generated[idx]['bleu'] = round(bleu.score,2)
        if no_ref == '' or no_pred == '':
            continue
        
        bleu_reduced = BLEU().corpus_score([no_pred], [[no_ref]])
        dict_generated[idx]['bleu_reduced'] = round(bleu_reduced.score,2)
        
        preds.append(dict_generated[idx]['pred'])
        refs.append(dict_generated[idx]['ref'])
        preds_reduced.append(dict_generated[idx]['no_pred'])
        refs_reduced.append(dict_generated[idx]['no_ref'])
    
    bleu = BLEU().corpus_score(preds, [refs])
    bleu1 = BLEU(max_ngram_order=1).corpus_score(preds, [refs])
    bleu2 = BLEU(max_ngram_order=2).corpus_score(preds, [refs])
    bleu3 = BLEU(max_ngram_order=3).corpus_score(preds, [refs])
    bleu4 = BLEU(max_ngram_order=4).corpus_score(preds, [refs])
    reduced_bleu = BLEU().corpus_score(preds_reduced, [refs_reduced])
    
    chrf = CHRF(word_order=2).corpus_score(preds, [refs])
    reduced_chrf = CHRF(word_order=2).corpus_score(preds_reduced, [refs_reduced])
    return bleu, bleu1, bleu2, bleu3, bleu4, reduced_bleu, chrf, reduced_chrf
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text generation output and compute BLEU scores from task generate output.')
    parser.add_argument('--generates-dir', type=str, required=True, help='Path where the generates are located. Before the experiment/generates/partition folders.')
    parser.add_argument('--vocab-dir', type=str, required=True, help='Path where the vocabulary is stored. Before the /vocab folder')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name that we want to test.')
    parser.add_argument('--partition', type=str, required=True, help='Partition: test/val/test')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint name that we want to test, without the extension. For example: checkpoint_best')
    args = parser.parse_args()
    
    generate_file = f'{args.generates_dir}/{args.experiment}/generates/cvpr23.fairseq.i3d.{args.partition}.how2sign/{args.checkpoint}.out'
    blacklist_file = 'scripts/blacklisted_words.txt'
    path_to_vocab = f'{args.vocab_dir}/vocab/cvpr23.train.how2sign.unigram7000_lowercased.model'
    print(f'Analyzing file: {generate_file}', flush=True)
    dict_generate = parse_generate_file(generate_file, blacklist_file, args.partition, path_to_vocab)
    bleu, bleu_1, bleu_2, bleu_3, bleu_4, reduced_bleu, chrf, reduced_chrf = compute_metrics(dict_generate)
    print(f"For experiment: {args.experiment} for cvpr23.fairseq.i3d.{args.partition}.how2sign")
    print("BLEU: ", bleu)
    print(f"bleu_1: {bleu_1.score}, bleu_2: {bleu_2.score}, bleu_3: {bleu_3.score}, bleu_4: {bleu_4.score}")
    print("Reduced BLEU: ", reduced_bleu)
    print("CHRF: ", chrf)
    print("Reduced CHRF: ", reduced_chrf)