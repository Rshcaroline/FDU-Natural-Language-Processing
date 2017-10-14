import math, collections

DISCOUNT = 1

def bigram_at(sentence, i):
  w1 = sentence[i].word
  w2 = sentence[i+1].word
  return '%s %s' % (w1, w2)

# Kneser-Kney Smoothing Algorithm
class CustomLanguageModel:

  # Initialize your data structures in the constructor.
  def __init__(self, corpus):
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.unigram_total = 0
    
    self.bigramCounts = collections.defaultdict(lambda: 0)

    self.afterWordCounts = collections.defaultdict(lambda: 0)
    self.beforWordCounts = collections.defaultdict(lambda: 0)

    self.train(corpus)


  # Takes a HolbrookCorpus corpus, does whatever training is needed.
  def train(self, corpus):
    # --- UNIGRAM & BIGRAM COUNTS ---------------

    for sentence in corpus.corpus:
      datums = sentence.data
      # Populate unigram counts & total
      for datum in datums:
        token = datum.word
        self.unigramCounts[token] += 1
        self.unigram_total += 1

      # Populate bigram counts & total
      for i in range(0, len(datums) - 1):  # ignore </s> at end
        token = bigram_at(datums, i)
        self.bigramCounts[token] += 1

    # Laplace smoothing of unigram (avoid 0 in denominator for unseens later)
    self.unigramCounts['UNK'] = 0
    for token in self.unigramCounts:
      self.unigramCounts[token] += 1
      self.unigram_total += 1

    for unigram in self.unigramCounts:
      self.afterWordCounts[unigram] = count_followers(unigram, self.bigramCounts)
      self.beforWordCounts[unigram] = count_preceding(unigram, self.bigramCounts)

  ##
  # Takes a list of strings, returns a log-probability score of that
  # sentence using data from train().
  def score(self, sentence):
    score = 0.0

    for i in range(0, len(sentence) - 1):  # ignore </s> at end
      bigram = '%s %s' % (sentence[i], sentence[i+1])
      bigram_count = self.bigramCounts[bigram]
      w1_unigram = sentence[i] if (self.unigramCounts[sentence[i]] > 0) else 'UNK'
      w2_unigram = sentence[i+1]
      # Force floating-point division
      discount_bigram = max(bigram_count - DISCOUNT, 0)               \
                        / (self.unigramCounts[w1_unigram] * 1.0)

      interpolation_weight = calc_interpoltn_weight(w1_unigram,           \
                                                    self.unigramCounts,   \
                                                    self.afterWordCounts)
      w2_contin_prob = calc_contin_prob(w2_unigram,             \
                                        self.beforWordCounts,   \
                                        self.bigramCounts)

      KN_prob = discount_bigram + interpolation_weight * w2_contin_prob

      # print '%f %f %f' % (discount_bigram, interpolation_weight, w2_contin_prob)
      score += math.log(KN_prob+.00000001)

    return score


def calc_interpoltn_weight(w1_unigram, unigramCounts, afterWordCounts):
  normalized_discount = DISCOUNT / (unigramCounts[w1_unigram] * 1.0)
  n_followers = afterWordCounts[w1_unigram]
  return normalized_discount * n_followers


def calc_contin_prob(w2, beforWordCounts, bigramCounts):
                                          # Force floating-point division
  return (beforWordCounts[w2] * 1.0) / (len(bigramCounts) * 2.0)


def count_followers(w1_unigram, bigramCounts):
  n_followers = 0
  for bigram in bigramCounts:
    if bigram.startswith(w1_unigram):   n_followers += 1
  return n_followers


def count_preceding(w2_unigram, bigramCounts):
  n_preceding = 0
  for bigram in bigramCounts:
    if bigram.endswith(w2_unigram):   n_preceding += 1
  return n_preceding