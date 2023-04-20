import pandas as pd
import argparse
from pathlib import Path
import string
from collections import Counter
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-in", required=True, type=str) #/mnt/gpid08/users/ltarres/CVPR23_experiments/cvpr23.fairseq.mediapipe.train.how2sign.tsv
    parser.add_argument("--column", required=True, default = 'translation', type=str) #dataframe.columns.values: ['id', 'translation_tokenized', 'signs_file', 'signs_offset', 'signs_length', 'signs_type', 'signs_lang', 'translation', 'translation_lang', 'glosses', 'topic', 'signer_id']
    parser.add_argument("--top-frequent", required=True, default = 10, type=int)
    parser.add_argument("--plot-histogram", default = True, type=bool)
    parser.add_argument("--save-list", default = True, type=bool)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    tsv_in = Path(args.tsv_in).expanduser().resolve() 
    dataframe = pd.read_csv(tsv_in, sep='\t', header=0) 
    
    #Have a with all the words
    all_words = []
    for line in dataframe[args.column]:
        #remove punctuation from line and split with spaces
        line = line.lower()
        all_words.extend(line.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')).split(' '))
    
    word_frequency = Counter(all_words).most_common(args.top_frequent)
    words = [word for word, _ in word_frequency]
    counts = [counts for _, counts in word_frequency]
    
    print(f"Top {args.top_frequent} most frequent words train data: {words}")
    
    if args.plot_histogram:
        plt.bar(words, counts)
        plt.xticks(rotation='vertical', fontsize=3)
        plt.title(f"{args.top_frequent} most frequent words in training")
        plt.ylabel("Frequency")
        plt.xlabel("Words")
        plt.show()
        plt.tight_layout()
        plt.savefig(f"{args.top_frequent}_most_frequent_words_in_train_data.png")
    
    if args.save_list:
        with open(f"top_{args.top_frequent}_most_frequent_words_lower.txt", 'w') as f:
            for word in words:
                f.write(str(word)+'\n')
    
if __name__ == "__main__":
    main()