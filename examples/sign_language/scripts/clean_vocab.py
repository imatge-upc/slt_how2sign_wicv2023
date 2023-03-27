#This file cleans and normalizes the how2sign vocabulary present in the .tsv files

import re

from sacremoses import MosesTokenizer
import contractions

from spellchecker import SpellChecker
from tqdm import tqdm

def load_h2s(path):
    with open(path, "r") as file:
        data = file.read().splitlines()

    data = [line.split('\t') for line in data]
    return data

#Maybe we can do something like this: https://github.com/Joee1995/eng_text_norm/blob/master/cleaners.py
#With the issues that we have: convert to ascii
#replace symbolic expressions
#normalize operators
#normalize punctuation


#I will need to pass this as a parameter
sentence_path = {'val': "/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.fairseq.mediapipe.val.how2sign.tsv",
                      'test': "/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.fairseq.mediapipe.test.how2sign.tsv",
                      'train': "/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.fairseq.mediapipe.train.how2sign.tsv"
                }
#TODO: Check the paths, to see if they are the final ones, and aligned ones that I am using in fairseq.

train = load_h2s(sentence_path["train"])
val = load_h2s(sentence_path["val"])
test = load_h2s(sentence_path["test"])


def check_spelling(sentence, sent_id, incorrect_sentences):
    sentence = sentence.replace("--", " - ")
    spell = SpellChecker()
    #Check for misspelled words, 
    # Things missinng: 
    # this weird thing: &apos;s, why? substitute for is. 
    # numbers 1, 2, 3.. we also need to substitute them for the word. -> using the library num2words
    misspelled = spell.unknown(sentence.split())
    correction = {}
    for word in misspelled:
        correction[word] = spell.correction(word)
    
    if len(correction) > 0:
        incorrect_sentences.append(sent_id)

    correct_sent = sentence
    print(f'correction: {correction}')
    for word in correction:
        if correction[word] == None: #If the word is not in the dictionary, it returns None. We just puth the word back
            correction[word] = word
        correct_sent = correct_sent.replace(word, correction[word])
    return correct_sent

def sentence_preprocess(data):
    mt = MosesTokenizer(lang='en')
    incorrect_sentences = []
    
    whatever = []
    tokenized_list = []
    for sentence in tqdm(data[1:]):
        words_fixed = []
        sentence = sentence[-1].lower()
        print(f'sentence: {sentence}')
        sent_id = sentence[2]
        decontracted_words = []
        for word in re.findall(r"[\w']+|[.,!?;-]", check_spelling(sentence, sent_id, incorrect_sentences)):
            decontracted = contractions.fix(word)
            #num2words(number)
            decontracted_words.append()
            print(f'decontracted_words: {decontracted_words}')
        whole_sentence = ' '.join(decontracted_words)
        print(f'whole_sentence: {whole_sentence}')
        tokenized = mt.tokenize(whole_sentence, return_str=True)
        print(f'tokenized: {tokenized}')
        tokenized_list.append(tokenized)

    whatever = '\n'.join(tokenized_list)
    return whatever, incorrect_sentences
    
    '''
    return '\n'.join([mt.tokenize(' '.join([contractions.fix(word) 
                          for word in re.findall(r"[\w']+|[.,!?;-]", check_spelling(sentence[-5].lower(), sentence[0], incorrect_sentences))]), 
                                  return_str=True)
                      for sentence in tqdm(data[1:])]), incorrect_sentences
    '''

val_prep, inc_sent_val = sentence_preprocess(val)
breakpoint()
#test_prep, inc_sent_test = sentence_preprocess(test)
#train_prep, inc_sent_train = sentence_preprocess(train)

#assert len(val[1:]) == len(val_prep.split('\n'))
#assert len(test[1:]) == len(test_prep.split('\n'))
#assert len(train[1:]) == len(train_prep.split('\n'))



#df = pd.DataFrame([('Foreign Cinema', 'Restaurant', 289.0),('Liho Liho', 'Restaurant', 224.0),('500 Club', 'bar', 80.5),('The Square', 'bar', 25.30)],columns=('name', 'type', 'AvgBill'))
#df.insert(loc=1, column="Stars", value=[2,2,3,4])

#Add a new column with this data. check that all the id's are the same (and sentences have been corrected)
current_header = ['id','signs_file','signs_offset','signs_length', 'signs_type', 'signs_lang', 
                  'translation', 'translation_lang', 'glosses', 'topic', 'signer_id']
df=pd.read_csv(sentence_path_laia['val'], sep='\t',names=current_header)

df.insert(
    loc=7, #this should be after translation
    column='translation_corrected',
    value=val_prep)


#The pipeline of reconstruction is:
#i guess ignore the contractions:

#Recovering punctuation:
#pip install distilbert-punctuator
from dbpunctuator.inference import Inference, InferenceArguments
from dbpunctuator.utils import DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP

args = InferenceArguments(model_name_or_path="Qishuai/distilbert_punctuator_en", tokenizer_name="Qishuai/distilbert_punctuator_en", tag2punctuator=DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP)
punctuator_model = Inference(inference_args=args, verbose=False)

text = [""" however when I am elected I vow to protect our American workforce unlike my opponent I have faith in our perseverance our sense of trust and our democratic principles will you support me"""]
text = [""" i want to go to new york"""] #It does not capitalize the rest of the words, but it does add the punctuation
print(punctuator_model.punctuation(text)[0])

#True Casing:
#pip install truecase
#pip install nltk
import truecase
import nltk
nltk.download('punkt')
truecase.get_true_case('hey, what is the weather in new york?')


