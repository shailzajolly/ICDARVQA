from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import sys
import csv
import json
csv.field_size_limit(sys.maxsize)

import pickle

import numpy as np
import nltk
nltk.data.path.append('data')
nltk.download('punkt', download_dir='data')
from tqdm import tqdm

from constants import *

data_path = os.path.join('data', 'train_task_1.json')
glove_path = os.path.join('data', 'glove', 'glove.6B.300d.txt')


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ';', r"/", '[', ']', '"', '{', '}',
    '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!'
]


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
        or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def process_txt(str1):
    return process_digit_article(process_punctuation(str1))


def process_a(freq_thr=9):

    train_data = json.load(open(data_path))['data']

    print("Calculating the frequency of each multiple choice answer...")
    ans_freqs = {}
    for item in tqdm(train_data):
        answers = item['dictionary']
        for ans in answers:
            temp_ans = process_txt(ans)
            ans_freqs[temp_ans] = ans_freqs.get(temp_ans, 0) + 1

    # filter out rare answers
    for a, freq in list(ans_freqs.items()):
        if freq < freq_thr:
            ans_freqs.pop(a)

    print("Number of answers appear more than {} times: {}".format(freq_thr - 1, len(ans_freqs)))

    # generate answer dictionary
    idx2ans = ['PAD', 'UNK', 'SOS', 'EOS']
    ans2idx = {'PAD': PAD_TOKEN, 'UNK': UNK_TOKEN, 'SOS': SOS_TOKEN, 'EOS': EOS_TOKEN}
    for i, a in enumerate(ans_freqs):
        idx2ans.append(a)
        ans2idx[a] = len(idx2ans) - 1

    targets = []
    for item in tqdm(train_data):
        for ans in item['answers']:
            targets.append({
                'question_id': item['question_id'],
                'file_path': item['file_path'].split('.')[0],
                'answer': process_txt(ans)
            })

    pickle.dump([idx2ans, ans2idx], open(os.path.join('data', 'dict_ans.pkl'), 'wb'))
    return targets, idx2ans


def process_qa(targets, max_words=14):

    print("Merging QAs...")
    idx2word = ['PAD', 'UNK', 'SOS', 'EOS']
    word2idx = {'PAD': PAD_TOKEN, 'UNK': UNK_TOKEN, 'SOS': SOS_TOKEN, 'EOS': EOS_TOKEN}

    train_data = json.load(open(data_path))['data']
    ques_ans = []
    counter = 0
    for i, item in enumerate(tqdm(train_data)):
        tokens = [i.lower() for i in item['question_tokens']]
        for token in tokens:
            if token not in word2idx:
                idx2word.append(token)
                word2idx[token] = len(idx2word) - 1

        assert item['question_id'] == targets[counter]['question_id'],\
                "Question ID doesn't match ({}: {})".format(item['question_id'],
                                                            targets[counter]['question_id'])

        for _ in range(len(item['answers'])):
            ques_ans.append({
                'file_path': item['file_path'].split('.')[0],
                'question': item['question'],
                'question_id': item['question_id'],
                'question_toked': tokens,
                'answer': targets[counter]['answer'],
                'answers': [process_txt(ans) for ans in item['answers']],
                'dictionary': [process_txt(ans) for ans in item['dictionary']]
            })
            counter += 1

    pickle.dump([idx2word, word2idx], open(os.path.join('data', 'dict_q.pkl'), 'wb'))
    pickle.dump(ques_ans, open(os.path.join('data', 'data_qa.pkl'), 'wb'))

    return idx2word


def process_wemb(idx2word, embed_type):
    print("Generating pretrained word embedding weights...")
    word2emb = {}
    emb_dim = int(glove_path.split('.')[-2].split('d')[0])
    #with open(glove_path) as f:
    f = open(glove_path, 'r+', encoding="utf-8")
    for entry in f:
        vals = entry.split(' ')
        word = vals[0].lower()
        word2emb[word] = np.asarray(vals[1:], dtype=np.float32)

    pretrained_weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        pretrained_weights[idx] = word2emb[word]

    np.save(os.path.join('data', 'glove_pretrained_{}.npy'.format(embed_type)),
            pretrained_weights)


if __name__ == '__main__':
    targets, idx2ans = process_a()
    idx2word = process_qa(targets)
    process_wemb(idx2word, 'question')
    process_wemb(idx2ans, 'answer')
    print("Done")

