import math, collections

# Language model that uses uniform probabilities for all words."""
class UniformLanguageModel:

  # Initialize your data structures in the constructor."""
  def __init__(self, corpus):
    self.words = set([])
    self.train(corpus)

  ##
  # Takes a corpus and trains your language model.
  # Compute any counts or other corpus statistics in this function.
  def train(self, corpus):
    for sentence in corpus.corpus: # iterate over sentences in the corpus
      for datum in sentence.data: # iterate over datums in the sentence
        word = datum.word # get the word
        self.words.add(word)

  ##
  # Takes a list of strings as argument and returns the log-probability
  # of the sentence using your language model. Use whatever data you
  # computed in train() here.
  def score(self, sentence):

    score = 0.0
    probability = math.log(1.0/len(self.words))
    for token in sentence: # iterate over words in the sentence
      score += probability
    # NOTE: a simpler method would be just score = sentence.size() * - Math.log(words.size()).
    # we show the 'for' loop for insructive purposes.
    return score

