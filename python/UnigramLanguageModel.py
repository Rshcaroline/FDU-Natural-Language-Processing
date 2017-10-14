import math, collections

# Language model that uses unigram probabilities, ignoring unseen words.
class UnigramLanguageModel:

  def __init__(self, corpus):
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  # Takes a HolbrookCorpus corpus, does whatever training is needed.
  def train(self, corpus):
    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        self.unigramCounts[token] = self.unigramCounts[token] + 1
        self.total += 1
  
  # Takes a list of strings, returns a score of that sentence.
  def score(self, sentence):
    score = 0.0 
    for token in sentence:
      count = self.unigramCounts[token]
      if count > 0:
        # Must add and subtract scores b/c **logs** of the scores
        score += math.log(count) - math.log(self.total)
      # Ignore unseen words
    return score
