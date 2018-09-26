import torch

from collections import defaultdict


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = defaultdict(int)
        self.total = 0

    def addWord(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        tokenId = self.word2idx[word]
        self.counter[tokenId] += 1
        self.total += 1
        return self.word2idx[word]

    def orderByFrequency(self):
        self.idx2word = []
        self.word2idx = {}
        wordsAndCounts = sorted(
            list(self.counter.items()), key=lambda x: x[1], reverse=True)
        wordsByCount = [x[0] for x in wordsAndCounts]
        for index, word in enumerate(wordsByCount):
            self.idx2word.append(word)
            self.word2idx[word] = index

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, trainFolder, validFolder, testFolder, limit=-1):
        self.dictionary = Dictionary()
        self.limit = limit
        self.train_positive = self.tokenize(trainFolder + 'positive.txt')
        self.train_negative = self.tokenize(trainFolder + 'negative.txt')
        self.valid_positive = self.tokenize(validFolder + 'positive.txt')
        self.valid_negative = self.tokenize(validFolder + 'negative.txt')
        self.test_positive = self.tokenize(testFolder + 'positive.txt')
        self.test_negative = self.tokenize(testFolder + 'negative.txt')
        # self.dictionary.orderByFrequency()

    def tokenize(self, path):
        """Tokenizes a text file."""
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for index, line in enumerate(f):
                if self.limit > 0 and index > self.limit:
                    break
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.addWord(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = []
            for index, line in enumerate(f):
                currIds = []
                if self.limit > 0 and index > self.limit:
                    break
                words = line.split() + ['<eos>']
                for word in words:
                    currIds.append(self.dictionary.word2idx[word])
                ids.append(torch.LongTensor(currIds))

        return ids


def batchify(data, batch_size, cuda):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, batch_num, seq_len, evaluation=False):
    seq_len = min(seq_len, len(source) - 1 - batch_num)
    data = source[batch_num:batch_num+seq_len]
    target = source[batch_num+1:batch_num+1+seq_len].view(-1)
    return data, target
