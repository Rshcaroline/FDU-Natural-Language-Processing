import math, collections


def bigram_at(sentence, i):
  w1 = sentence[i].word
  w2 = sentence[i+1].word
  return '%s %s' % (w1, w2)


class StupidBackoffLanguageModel:

  # Initialize your data structures in the constructor.
  def __init__(self, corpus):
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.unigram_total = 0
    
    self.bigramCounts = collections.defaultdict(lambda: 0)

    self.train(corpus)


  # Takes a HolbrookCorpus corpus, does whatever training is needed.
  def train(self, corpus):
    # --- UNIGRAM & BIGRAM COUNTS ---------------

    for sentence in corpus.corpus:
      # Populate unigram counts & total
      for datum in sentence.data:
        token = datum.word
        self.unigramCounts[token] += 1
        self.unigram_total += 1

      # Populate bigram counts & total
      for i in range(0, len(sentence.data) - 1):  # ignore </s> at end
        token = bigram_at(sentence.data, i)
        self.bigramCounts[token] += 1


    # --- LAPLACE SMOOTHING ---------------
    # Increment each token & the unigram_total by 1

    ### UNIGRAM LAPLACE SMOOTHING ###
    self.unigramCounts['UNK'] = 0
    for token in self.unigramCounts:
      self.unigramCounts[token] += 1
      self.unigram_total += 1

    ### BIGRAM LAPLACE SMOOTHING ###
    self.bigramCounts['UNK'] = 0
    for token in self.bigramCounts:
      self.bigramCounts[token] += 1


  ##
  # Takes a list of strings, returns a log-probability score of that
  # sentence using data from train().
  def score(self, sentence):
    score = 0.0

    for i in range(0, len(sentence) - 1):  # ignore </s> at end
      bigram_token = '%s %s' % (sentence[i], sentence[i+1])
      bigram_count = self.bigramCounts[bigram_token]

      prev_word = sentence[i]
      prev_word_count = self.unigramCounts[prev_word]

      if bigram_count > 0:
        score += math.log(bigram_count) - math.log(prev_word_count)
      else:
        unigram_count = self.unigramCounts[sentence[i+1]]
        score += math.log(unigram_count + 1) -        \
                 math.log(self.unigram_total) +       \
                 math.log(0.4)

    return score