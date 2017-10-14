import json
import urllib
import urllib2
import hashlib
import random
import email
import email.message
import email.encoders

from shared_submit import Submitter

from EditModel import EditModel
from HolbrookCorpus import HolbrookCorpus
from UnigramLanguageModel import UnigramLanguageModel
from StupidBackoffLanguageModel import StupidBackoffLanguageModel
from LaplaceUnigramLanguageModel import LaplaceUnigramLanguageModel
from LaplaceBigramLanguageModel import LaplaceBigramLanguageModel
from CustomLanguageModel import CustomLanguageModel
from SpellCorrect import SpellCorrect

class AutocorrectSubmitter(Submitter):

  def validParts(self):
    """Returns a list of valid part names."""

    partNames = [ 'Edit Model Dev', \
                  'Edit Model Test', \
                  'Laplace Unigram Dev', \
                  'Laplace Unigram Test', \
                  'Laplace Bigram Dev', \
                  'Laplace Bigram Test', \
                  'Stupid Backoff Dev', \
                  'Stupid Backoff Test', \
                  'Custom Dev', \
                  'Custom Test'
                ]
    return partNames


  def sources(self):
    """Returns source files, separated by part. Each part has a list of files."""
    srcs = [
             [ 'EditModel.py' ], \
             [ 'EditModel.py' ], \
             [ 'LaplaceUnigramLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'], \
             [ 'LaplaceUnigramLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'], \
             [ 'LaplaceBigramLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'], \
             [ 'LaplaceBigramLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'], \
             [ 'StupidBackoffLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'], \
             [ 'StupidBackoffLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'], \
             [ 'CustomLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'], \
             [ 'CustomLanguageModel.py', 'SpellCorrect.py', 'EditModel.py'],
           ]
    return srcs


  def homework_id(self):
    """Returns the string homework id."""
    return '2'

  def output(self, partId, ch_aux):
    """Uses the student code to compute the output for test cases."""
    trainCorpus = HolbrookCorpus('../data/holbrook-tagged-train.dat')

    if partId in [1,2]:
      editModel = EditModel('../data/count_1edit.txt', trainCorpus)
      return json.dumps([[(e.editedWord, e.rule()) for e in editModel.edits(line.strip())] for line in ch_aux.split("\n")])
    else:
      testCorpus = HolbrookCorpus()
      testCorpus.slurpString(ch_aux)
      lm = None
      if partId in [3,4]:
        lm = LaplaceUnigramLanguageModel(trainCorpus)
      elif partId in [5,6]:
        lm = LaplaceBigramLanguageModel(trainCorpus)
      elif partId in [7,8]:
        lm = StupidBackoffLanguageModel(trainCorpus)
      elif partId in [9,10]:
        lm = CustomLanguageModel(trainCorpus)
      else:
        print 'Unknown partId: " + partId'
        return None

      speller = SpellCorrect(lm, trainCorpus)
      output = speller.correctCorpus(testCorpus)
      # put in the part ID as well
      output = '[["%d"],%s' % (partId, output[1:])
      return output

submitter = AutocorrectSubmitter()
submitter.submit(0)
