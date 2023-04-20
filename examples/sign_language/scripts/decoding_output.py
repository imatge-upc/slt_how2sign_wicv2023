#!/usr/bin/env python
import sys
import truecase
from sacremoses import MosesDetokenizer

def decode(text, moses_detok):
    # Perform custom decoding here
    #s = self.bpe_tokenizer.decode(text) in tthe generate I have this line --remove-bpe that is supposed to do the decoding
    decoded_text = moses_detok.detokenize(text.split())
    decoded_text = truecase.get_true_case(decoded_text)
    return decoded_text

def main():
    #--post-process "python scripts/decoding_output.py" \ -> this does not work :(
    import pdb; pdb.set_trace()
    moses_detok = MosesDetokenizer(lang='en')
    for line in sys.stdin:
        decoded_line = decode(line.strip(), moses_detok)
        sys.stdout.write(decoded_line + '\n')
        sys.stdout.flush()

if __name__ == '__main__':
    main()