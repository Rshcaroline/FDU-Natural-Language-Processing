import math, collections


def bigram_at(sentence, i):
  w1 = sentence[i].word
  w2 = sentence[i+1].word
  return '%s %s' % (w1, w2)


class LaplaceBigramLanguageModel:

  # Initialize your data structures in the constructor.
  def __init__(self, corpus):
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)


  # Takes a HolbrookCorpus corpus, does whatever training is needed.
  def train(self, corpus):
    for sentence in corpus.corpus:
      for i in range(0, len(sentence.data) - 1):  # ignore </s> at end
        token = bigram_at(sentence.data, i)
        self.bigramCounts[token] += 1
        self.total += 1

    self.bigramCounts['UNK'] = 0

    # For each token, increment by 1 for Laplace smoothing
    for token in self.bigramCounts:
      self.bigramCounts[token] += 1
      self.total += 1


  ##
  # Takes a list of strings, returns a log-probability score of that
  # sentence using data from train().
  def score(self, sentence):
    score = 0.0 
    for i in range(0, len(sentence) - 1):  # ignore </s> at end
      token = '%s %s' % (sentence[i], sentence[i+1])
      count = self.bigramCounts[token] | self.bigramCounts['UNK']
      if count > 0:
        # Must add and subtract scores b/c **logs** of the scores
        score += math.log(count) - math.log(self.total)
      # Ignore unseen words
    return score