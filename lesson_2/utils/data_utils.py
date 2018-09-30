import torch

from collections import defaultdict


class CharDictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = defaultdict(int)
        self.total = 0

    def addChar(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        tokenId = self.char2idx[char]
        self.counter[tokenId] += 1
        self.total += 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)


class CharCorpus(object):
    def __init__(
            self, trainFile1, trainFile2, validFile1, validFile2,
            testFile1, testFile2, limit=-1):
        self.dictionary = CharDictionary()
        self.limit = limit
        self.train_1 = self.tokenize(trainFile1)
        self.train_2 = self.tokenize(trainFile2)
        self.valid_1 = self.tokenize(validFile1)
        self.valid_2 = self.tokenize(validFile2)
        self.test_1 = self.tokenize(testFile1)
        self.test_2 = self.tokenize(testFile2)
        # self.dictionary.orderByFrequency()

    def tokenize(self, path):
        """Tokenizes a text file."""
        # Add words to the dictionary
        self.dictionary.addChar('<eos>')
        with open(path, 'r') as f:
            tokens = 0
            for index, line in enumerate(f):
                if self.limit > 0 and index > self.limit:
                    break
                chars = line
                tokens += len(chars)
                for char in chars:
                    self.dictionary.addChar(char)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = []
            for index, line in enumerate(f):
                currIds = []
                if self.limit > 0 and index > self.limit:
                    break
                chars = list(line) + ['<eos>']
                for char in chars:
                    currIds.append(self.dictionary.char2idx[char])
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
