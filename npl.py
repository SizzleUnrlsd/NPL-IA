#!/usr/bin/env -S python3
import argparse
import chardet
import json
import os
import random
import re


class AI:
    def __init__(self, n=2, filepath='model.json'):
        self.n = n
        self.vocab = set()
        self.model = {}
        self.filepath = filepath
        if os.path.exists(self.filepath):
            self.load_model(self.filepath)

    def normalize(self, text):
        text = text.lower()
        text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace())
        text = ' '.join(text.split())
        return text

    def pre_tokenize(self, text):
        return text.split('.')

    def tokenize(self, sentence):
        return re.findall(r'\b\w+\b', sentence)

    def train(self, filepaths):
        for filepath in filepaths:
            if os.path.exists(filepath):
                rawdata = open(filepath, 'rb').read()
                result = chardet.detect(rawdata)
                encoding = result['encoding']

                with open(filepath, 'r', encoding=encoding) as file:
                    text = file.read()

                text = self.normalize(text)
                sentences = self.pre_tokenize(text)

                if not sentences:
                    print(f"No text to learn in file {filepath}")
                    continue

                for sentence in sentences:
                    tokens = self.tokenize(sentence)
                    self.vocab.update(tokens)

                    for i in range(len(tokens) - self.n):
                        gram = tuple(tokens[i:i + self.n])
                        if gram in self.model:
                            self.model[gram].append(tokens[i + self.n])
                        else:
                            self.model[gram] = [tokens[i + self.n]]

        if not self.vocab:
            print("No model or text provided for training. "
                  "Please provide a text or a model file for the AI to learn from.")
            return

        self.save_model(self.filepath)

    def predict(self, text, num_words=1):
        if not self.vocab:
            return "Please load texts or a valid json file"

        text = self.normalize(text)
        tokens = self.tokenize(text)
        if len(tokens) < self.n:
            return text

        result = tokens
        gram = tuple(tokens[-self.n:])

        for _ in range(num_words):
            if gram not in self.model:
                next_word = random.choice(list(self.vocab))
            else:
                possible_words = self.model[gram]
                next_word = random.choice(possible_words)

            result.append(next_word)
            gram = tuple((list(gram) + [next_word])[-self.n:])

        return ' '.join(result)

    def save_model(self, filepath):
        model_data = {
            'n': self.n,
            'vocab': list(self.vocab),
            'model': {' '.join(k): v for k, v in self.model.items()}
        }

        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                existing_data = json.load(file)
            model_data['model'].update(existing_data['model'])

        with open(filepath, 'w') as file:
            json.dump(model_data, file)

    def load_model(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                model_data = json.load(file)

            self.n = model_data['n']
            self.vocab = set(model_data['vocab'])
            self.model = {tuple(k.split(' ')): v for k, v in model_data['model'].items()}


def main():
    with open('config.json') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description=config.get('description', 'Default Description'))
    parser.add_argument('-n', '--num_words', type=int, default=config.get('num_words', 1), help='Number of words to generate')
    parser.add_argument('-s', '--start_sentence', type=str, default=config.get('default sentence', 'Les chats'), help='Start of the sentence to generate from')
    parser.add_argument('files', nargs='*', help='Text files to use for training the model')
    args = parser.parse_args()

    ai = AI()

    if args.files:
        ai.train(args.files)
    else:
        print("No text files provided for training. Using existing model if available.")

    print(ai.predict(args.start_sentence, args.num_words))


if __name__ == "__main__":
    main()
