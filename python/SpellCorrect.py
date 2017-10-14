import math
from Datum import Datum
from Sentence import Sentence
from HolbrookCorpus import HolbrookCorpus
from UniformLanguageModel import UniformLanguageModel
from UnigramLanguageModel import UnigramLanguageModel
from StupidBackoffLanguageModel import StupidBackoffLanguageModel
from LaplaceUnigramLanguageModel import LaplaceUnigramLanguageModel
from LaplaceBigramLanguageModel import LaplaceBigramLanguageModel
from CustomLanguageModel import CustomLanguageModel
from EditModel import EditModel
from SpellingResult import SpellingResult
import types

# Modified version of Peter Norvig's spelling corrector
"""
  Spelling Corrector.

  Copyright 2007 Peter Norvig. 
  Open source code under MIT license: http://www.opensource.org/licenses/mit-license.php
"""

import re, collections

##
# Spelling corrector for sentences. Holds edit model, language model
# and the corpus.
class SpellCorrect:

  def __init__(self, lm, corpus):
    self.languageModel = lm
    self.editModel = EditModel('../data/count_1edit.txt', corpus)

  ##
  # Assuming exactly one error per sentence, returns the most
  # probable corrected sentence. Sentence is a list of words.
  def correctSentence(self, sentence):
    # Tip: self.editModel.editProbabilities(word) gives edits and
      # log-probabilities according to your edit model. You should iterate
      # through these values instead of enumerating all edits.
      #
    # Tip: self.languageModel.score(trialSentence) gives log-probability
      # of a sentence

    if len(sentence) == 0:      return []

    bestSentence = sentence[:]   # copy of sentence
    bestScore = float('-inf')
    
    for i in xrange(1, len(sentence) - 1):   # ignore <s> and </s>
      #### Select max probability sentence, according to noisy channel model.

      edit_possibilities = self.editModel.editProbabilities(sentence[i])
      for edit in edit_possibilities:
        trialSentence = sentence[:i] + [edit[0]] + sentence[i+1:] 

        ##
        # The log-probabilities will come out as negative, because we're
          # taking the logs of very small numbers. Still, a larger log-probab.
          # is a "higher score". For instance, -3 indicates a higher probability
          # than -223802.
        # Must get sum (rather than product) of languageModel.score &
          # edit probability because they are logs.
        curr_score = self.languageModel.score(trialSentence) + edit[1]
        if (bestScore < curr_score):
          bestScore = curr_score
          bestSentence = trialSentence[:]

    # print sentence
    # print bestSentence
    # print bestScore
    # print ('\n----------\n')

    return bestSentence

  # Tests this speller on a corpus, returns a SpellingResult
  def evaluate(self, corpus):  
    numCorrect = 0
    numTotal = 0
    testData = corpus.generateTestCases()
    for sentence in testData:
      if sentence.isEmpty():  continue
      errorSentence = sentence.getErrorSentence()
      hypothesis = self.correctSentence(errorSentence)
      if sentence.isCorrection(hypothesis):   numCorrect += 1
      numTotal += 1
    return SpellingResult(numCorrect, numTotal)

  ##
  # Corrects a whole corpus, returns JSON representation of output.
  def correctCorpus(self, corpus): 
    string_list = []   # we will join these with commas,  bookended with []
    sentences = corpus.corpus
    for sentence in sentences:
      uncorrected = sentence.getErrorSentence()
      corrected = self.correctSentence(uncorrected)
      word_list = '["%s"]' % '","'.join(corrected)
      string_list.append(word_list)
    output = '[%s]' % ','.join(string_list)
    return output


##
# Trains all of the language models and tests them on the dev data.
  # Change devPath if you wish to do things like test on the training
  # data.
def main():
  trainPath = '../data/holbrook-tagged-train.dat'
  trainingCorpus = HolbrookCorpus(trainPath)

  devPath = '../data/holbrook-tagged-dev.dat'
  devCorpus = HolbrookCorpus(devPath)

  # print 'Unigram Language Model: ' 
  # unigramLM = UnigramLanguageModel(trainingCorpus)
  # unigramSpell = SpellCorrect(unigramLM, trainingCorpus)
  # unigramOutcome = unigramSpell.evaluate(devCorpus)
  # print str(unigramOutcome)

  # print 'Uniform Language Model: '
  # uniformLM = UniformLanguageModel(trainingCorpus)
  # uniformSpell = SpellCorrect(uniformLM, trainingCorpus)
  # uniformOutcome = uniformSpell.evaluate(devCorpus) 
  # print str(uniformOutcome)

  # print 'Laplace Unigram Language Model: ' 
  # laplaceUnigramLM = LaplaceUnigramLanguageModel(trainingCorpus)
  # laplaceUnigramSpell = SpellCorrect(laplaceUnigramLM, trainingCorpus)
  # laplaceUnigramOutcome = laplaceUnigramSpell.evaluate(devCorpus)
  # print str(laplaceUnigramOutcome)

  # print 'Laplace Bigram Language Model: '
  # laplaceBigramLM = LaplaceBigramLanguageModel(trainingCorpus)
  # laplaceBigramSpell = SpellCorrect(laplaceBigramLM, trainingCorpus)
  # laplaceBigramOutcome = laplaceBigramSpell.evaluate(devCorpus)
  # print str(laplaceBigramOutcome)

  # print 'Stupid Backoff Language Model: '  
  # sbLM = StupidBackoffLanguageModel(trainingCorpus)
  # sbSpell = SpellCorrect(sbLM, trainingCorpus)
  # sbOutcome = sbSpell.evaluate(devCorpus)
  # print str(sbOutcome)

  print 'Custom Language Model: '
  customLM = CustomLanguageModel(trainingCorpus)
  customSpell = SpellCorrect(customLM, trainingCorpus)
  customOutcome = customSpell.evaluate(devCorpus)
  print str(customOutcome)

if __name__ == "__main__":
    main()
