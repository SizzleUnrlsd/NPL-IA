# NPL-IA
Implementation of an n-gram model with text-based learning. It is used to complete the sequence of a sentence. 

# USAGE
usage: npl.py [-h] [-n NUM_WORDS] [-s START_SENTENCE] [files ...]

Train a basic n-gram model and use it to generate text.

positional arguments:
  files                 Text files to use for training the model

options:\n
\t-h, --help            show this help message and exit\n
\t-n NUM_WORDS, --num_words NUM_WORDS\n
                        Number of words to generate
\t-s START_SENTENCE, --start_sentence START_SENTENCE
                        Start of the sentence to generate from
