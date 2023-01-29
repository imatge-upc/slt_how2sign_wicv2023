"""
This code is to clean the vocabulary of the how2sign dataset.
Steps to follow:
1- lowercasing
2- check spelling
3- deal with contractions with Moses Tokenizer
4- remove punctuation

Things we have considered: keeping numbers as numbers, we will do truecasing and punctuation after the model has predicted the sentences. We should put this for inferece/test.
"""
from tqdm import tqdm
import pandas as pd

import re
#import contractions
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer, MosesTruecaser
from spellchecker import SpellChecker

from dbpunctuator.inference import Inference, InferenceArguments
from dbpunctuator.utils import DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP
import truecase
#import nltk
#nltk.download('punkt') #i think I have already done this once so it is not necessary anymore.


def load_h2s(path):
    df = pd.read_csv(path, sep='\t')
    return df

def check_spelling(sentence):
    spell = SpellChecker()
    misspelled = spell.unknown(sentence.split())
    
    correction = {}
    for word in misspelled:
        correction[word] = spell.correction(word)

    correct_sent = sentence
    for word in correction:
        if correction[word] == None: #If the word is not in the dictionary, it returns None. We just put the word back
            correction[word] = word
        correct_sent = correct_sent.replace(word, correction[word])
    return correct_sent
    
def clean_how2sign_vocabulary(path_to_data_sentencelevel):
    '''Pipeline for English eng_text_norm'''
    partition = ['train', 'val', 'test']
    print(f'loading: {path_to_data_sentencelevel[partition[0]]}')
    data = load_h2s(path_to_data_sentencelevel['train'])
    corrected_sentences = []
    
    #args = InferenceArguments(model_name_or_path="Qishuai/distilbert_punctuator_en", tokenizer_name="Qishuai/distilbert_punctuator_en", tag2punctuator=DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP)
    #punctuator_model = Inference(inference_args=args, verbose=False)
    
    mpn = MosesPunctNormalizer()
    mt = MosesTokenizer(lang='en')
    #mtr = MosesTruecaser('big.truecasemodel')
    md = MosesDetokenizer(lang='en')
    
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentence = row['translation'].lower() #lowercasing
        sent_id = row['id']
        print(f'Original sentence: {sentence}')
        print(f'Sentence id: {sent_id}')
        
        #check spelling
        #sentence = check_spelling(sentence) (I don't really like this, it makes some mistakes: hi -> his, cause -> because, aileron -> aileen)
        #print(f'Spell-checked sentence: {sentence}') #this already removes some punctuation
        
        #normalize input
        sentence = mpn.normalize(sentence)
        print(f'Sentence normalized: {sentence}')
        #remove contractions
        #sentence = contractions.fix(sentence, slang = False)
        #print(f'Without contractions sentence: {sentence}')
        sentence = mt.tokenize(sentence, return_str=True)
        print(f'Sentence tokenized,as we would input the model: {sentence}')
        
        #remove punctuation
        #sentence = sentence.replace("--", " ")
        #sentence = re.sub(r'[^\w\s]', '', sentence) #what to do if there is things like semi-circle, anti-itching, in-office, etc?
        #print(f'sentence as we would input the model: {sentence}')
        
        corrected_sentences.append(sentence)
        
        #To see how the reconstruction would look like
        #sentence = punctuator_model.punctuation([sentence])[0][0]
        #print(f'sentence adding synthetic punctuation: {sentence}')
        sentence = md.detokenize(sentence.split())
        print(f'sentence with detokenization: {sentence}')
        
        sentence = truecase.get_true_case(sentence)
        #sentence = mtr.truecase(sentence, return_str=True) #I really have no idea what I should do here...
        print(f'sentence with truecasing: {sentence}')
        
        #Instead of contractions, 
        #sentence = contractions.fix(sentence, slang = False)
        #print(f'sentence contracting: {sentence}')
    
    data.insert(loc=7, column="translation_tokenized", value=corrected_sentences)    
    data.to_csv(path_to_data_sentencelevel['train'], sep='\t', index=False)
    #I still need to do ctrl + c to quit the program        
    '''    
    text = convert_to_ascii(text) #TODO: check if this would be useful
    text = expand_abbreviations(text) #TODO: check if this would be useful
    text = remove_out_vocab(text)
    '''

if __name__ == "__main__":
    path_to_data_sentencelevel = {'val': "/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.fairseq.mediapipe.test.how2sign.tsv",\
                                  'test': "/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.fairseq.mediapipe.test.how2sign.tsv",\
                                  'train': "/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.fairseq.mediapipe.train.how2sign.tsv"\
    }
    clean_how2sign_vocabulary(path_to_data_sentencelevel)
    
'''
Something is not right with the training. Let's reproduce the steps:
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer
mpn = MosesPunctNormalizer()
mt = MosesTokenizer(lang='en')
mdt = MosesDetokenizer(lang='en')

#Bleu score needs to be computed against the original sentence, not the tokenized one
original_sentence = "And I call them decorative elements because basically all they're meant to do is to enrich and color the page."


lower_sentence = original_sentence.lower()
norm_sentence = mpn.normalize(lower_sentence)
ms_tok_sentence = mt.tokenize(norm_sentence, return_str=True) --> this should be the output to the model

#with the built vocabulary: /mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.train.how2sign.unigram7000_tokenized
#encode it so the model understands it
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceConfig
bpe_sentencepiece_model = '/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.train.how2sign.unigram7000_tokenized.model'

from fairseq.data import  encoders
from argparse import Namespace
bpe_tokenizer = encoders.build_bpe(Namespace(bpe='sentencepiece', sentencepiece_model= bpe_sentencepiece_model))

from pathlib import Path
dict_path = Path(bpe_sentencepiece_model).with_suffix('.txt')
if not dict_path.is_file():
    raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
from fairseq.data import Dictionary
tgt_dict = Dictionary.load(dict_path.as_posix())


sentence_bpe_tok = bpe_tokenizer.encode(ms_tok_sentence) #there should be no unknown tokens! 
#this sentence_model is what the model will actually see.            

#To compute the bleu we need to detokenize it:
sentence_bpe_detok = bpe_tokenizer.decode(sentence_bpe_tok)
sentence_moses_detok = mdt.detokenize(sentence_bpe_detok.split())

'''