import re
import math
import itertools
import collections
from HolbrookCorpus import HolbrookCorpus



class Edit(object):
  """Holder object for edits (and the rules used to generate them)."""
  
  def __init__(self, editedWord, corruptLetters, correctLetters):
    self.editedWord = editedWord
    # Represents x in the "P(x|w)" error probability term of the noisy channel model
    self.corruptLetters = corruptLetters
    # Represents w in the "P(x|w)" error probability term of the noisy channel model
    self.correctLetters = correctLetters

  def rule(self):
    return "%s|%s" % (self.corruptLetters, self.correctLetters)

  def __hash__(self):
    return hash(str(self))

  def __eq__(self, o):
    return str(self) == str(o)

  def __str__(self):
    return "Edit(editedWord=%s, rule=%s)" % (self.editedWord, self.rule())


##
# An object representing the edit model for a spelling correction task.
class EditModel(object):

  ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
  def __init__(self, editFile="../data/count_1edit.txt", corpus=None):
    if corpus:
      self.vocabulary = corpus.vocabulary()

    self.editCounts = {}
    with open(editFile) as f:
      for line in f:
        rule, countString = line.split("\t")
        self.editCounts[rule] = int(countString)
    
  ##
  # Returns a list of edits of 1-delete distance words and rules used to
  # generate them.
  def deleteEdits(self, word):
    
    if len(word) <= 0:      return []
    word = "<" + word   # Append start character

    result = []
    for i in xrange(1, len(word)):
      # The corrupted signal are this character and the character preceding
      corruptLetters = word[i-1:i+1] 
      # The correct signal is just the preceding character
      correctLetters = corruptLetters[:-1]

      # Corrected word deletes character i (& lacks start symbol <)
      correction = "%s%s" % (word[1:i], word[i+1:])
      result.append( Edit(correction, corruptLetters, correctLetters) )
      
    return result

  ##
  # Returns a list of edits of 1-insert distance words and rules used to
  # generate them.
  def insertEdits(self, word):
    # Tip: If inserting the letter 'a' as the second character in the word 'test', the corrupt
    #      signal is 't' and the correct signal is 'ta'. See slide 17 of the noisy channel model.
    #
    # Examples:   Edit(editedWord=ahi, rule=<|<a)
    #             Edit(editedWord=hai, rule=h|ha)
    #             Edit(editedWord=hia, rule=i|ia)

    if len(word) <= 0:      return []
    word = "<" + word    # Append start character

    result = []
    for i in xrange(1, len(word) + 1):
      for letter in EditModel.ALPHABET:
        # The corrupted signal is the character preceding
        corruptLetters = word[i-1] 
        # The correct signal is that preceding character + the new letter
        correctLetters = corruptLetters + letter

        # Corrected word inserts new letter (and lacks start symbol <)
        correction = "%s%s%s" % (word[1:i], letter, word[i:])
        result.append( Edit(correction, corruptLetters, correctLetters) )
      
    return result

  ##
  # Returns a list of edits of 1-transpose distance words and rules
  # used to generate them.
  def transposeEdits(self, word):
    # Tip: If tranposing letters 'te' in the word 'test', the corrupt signal is 'te'
    #      and the correct signal is 'et'. See slide 17 of the noisy channel model.
    #
    # Examples:   Edit(editedWord=ih, rule=hi|ih)

    result = []

    for i in xrange(1, len(word)):
      ##
      # The corrupted signal are this character and the preceding
      # character flipped
      corruptLetters = word[i-1] + word [i]
      correctLetters = corruptLetters[::-1]

      # The corrected word deletes character i (and lacks the start symbol)
      correction = "%s%s%s" % (word[0:i-1], correctLetters, word[i+1:])
      result.append( Edit(correction, corruptLetters, correctLetters) )

    return result

  ##
  # Returns a list of edits of 1-replace distance words and rules used to
  # generate them.
  def replaceEdits(self, word):
    # Tip: If replacing the letter 'e' with 'q' in the word 'test', the corrupt signal is 'e'
    #      and the correct signal is 'q'. See slide 17 of the noisy channel model.
    #
    # Examples:   Edit(editedWord=hu, rule=i|u)
    #             Edit(editedWord=ui, rule=h|u
    if len(word) <= 0:      return []

    result = []

    for i in xrange(0, len(word)):
      for letter in EditModel.ALPHABET:
        # Corrected word replaces character i with letter
        correction = word[:i] + letter + word[i+1:]

        ##
        # Only include true corrections (don't include cases where you replace
        # a letter with itself)
        if (correction != word):
          result.append( Edit(correction, word[i], letter) )
      
    return result



  ##
  # Returns a list of tuples of 1-edit distance words and rules used to
  # generate them, e.g. ("test", "te|et")
  def edits(self, word):
    #Note: this is just a suggested implementation, feel free to modify it for efficiency
    return  self.deleteEdits(word) + \
      self.insertEdits(word) + \
      self.transposeEdits(word) + \
      self.replaceEdits(word)

  ##
  # Computes in-vocabulary edits and edit-probabilities for a given
  # misspelling. Returns list of (correction, log(p(mispelling|correction)))
  # pairs.
  def editProbabilities(self, misspelling):

    wordCounts = collections.defaultdict(int)
    wordTotal  = 0
    for edit in self.edits(misspelling):
      if edit.editedWord != misspelling and edit.editedWord in self.vocabulary and edit.rule() in self.editCounts:
        ruleMass = self.editCounts[edit.rule()] 
        wordTotal += ruleMass
        wordCounts[edit.editedWord] += ruleMass


    #Normalize by wordTotal to make probabilities
    return [(word, math.log(float(mass) / wordTotal)) for word, mass in wordCounts.iteritems()]





### Start: Sanity checking code. ###


# Checks / prints the overlap between a guess and gold set."""
def checkOverlap(edits, gold):
  percentage = 100 * float(len(edits & gold)) / len(gold)
  missing = gold - edits
  extra = edits - gold
  for edit in edits:    print '- %s' % edit
  print "\tOverlap: %s%%" % percentage
  print "\tMissing edits: %s" % map(str, missing)
  print "\tExtra edits: %s" % map(str, extra)


# Sanity checks the edit model on the word 'hi'.
def main():

  trainPath = '../data/holbrook-tagged-train.dat'
  trainingCorpus = HolbrookCorpus(trainPath)
  editModel = EditModel("../data/count_1edit.txt", trainingCorpus)
  #These are for testing, you can ignore them
  DELETE_EDITS = set(['Edit(editedWord=i, rule=<h|<)', 'Edit(editedWord=h, rule=hi|h)'])
  INSERT_EDITS = set([
    Edit('ahi','<','<a'),Edit('bhi','<','<b'),Edit('chi','<','<c'),Edit('dhi','<','<d'),Edit('ehi','<','<e'),Edit('fhi','<','<f'),Edit('ghi','<','<g'),Edit('hhi','<','<h'),Edit('ihi','<','<i'),Edit('jhi','<','<j'),Edit('khi','<','<k'),Edit('lhi','<','<l'),Edit('mhi','<','<m'),Edit('nhi','<','<n'),Edit('ohi','<','<o'),Edit('phi','<','<p'),Edit('qhi','<','<q'),
    Edit('rhi','<','<r'),Edit('shi','<','<s'),Edit('thi','<','<t'),Edit('uhi','<','<u'),Edit('vhi','<','<v'),Edit('whi','<','<w'),Edit('xhi','<','<x'),Edit('yhi','<','<y'),Edit('zhi','<','<z'),Edit('hai','h','ha'),Edit('hbi','h','hb'),Edit('hci','h','hc'),Edit('hdi','h','hd'),Edit('hei','h','he'),Edit('hfi','h','hf'),Edit('hgi','h','hg'),Edit('hhi','h','hh'),
    Edit('hii','h','hi'),Edit('hji','h','hj'),Edit('hki','h','hk'),Edit('hli','h','hl'),Edit('hmi','h','hm'),Edit('hni','h','hn'),Edit('hoi','h','ho'),Edit('hpi','h','hp'),Edit('hqi','h','hq'),Edit('hri','h','hr'),Edit('hsi','h','hs'),Edit('hti','h','ht'),Edit('hui','h','hu'),Edit('hvi','h','hv'),Edit('hwi','h','hw'),Edit('hxi','h','hx'),Edit('hyi','h','hy'),Edit('hzi','h','hz'),
    Edit('hia','i','ia'),Edit('hib','i','ib'),Edit('hic','i','ic'),Edit('hid','i','id'),Edit('hie','i','ie'),Edit('hif','i','if'),Edit('hig','i','ig'),Edit('hih','i','ih'),Edit('hii','i','ii'),Edit('hij','i','ij'),Edit('hik','i','ik'),Edit('hil','i','il'),Edit('him','i','im'),Edit('hin','i','in'),Edit('hio','i','io'),Edit('hip','i','ip'),Edit('hiq','i','iq'),Edit('hir','i','ir'),
    Edit('his','i','is'),Edit('hit','i','it'),Edit('hiu','i','iu'),Edit('hiv','i','iv'),Edit('hiw','i','iw'),Edit('hix','i','ix'),Edit('hiy','i','iy'),Edit('hiz','i','iz')
  ])
  TRANPOSE_EDITS = set([Edit('ih','hi','ih')])
  REPLACE_EDITS = set([
    Edit('ai','h','a'),Edit('bi','h','b'),Edit('ci','h','c'),Edit('di','h','d'),Edit('ei','h','e'),Edit('fi','h','f'),Edit('gi','h','g'),Edit('ii','h','i'),Edit('ji','h','j'),
    Edit('ki','h','k'),Edit('li','h','l'),Edit('mi','h','m'),Edit('ni','h','n'),Edit('oi','h','o'),Edit('pi','h','p'),Edit('qi','h','q'),Edit('ri','h','r'),Edit('si','h','s'),Edit('ti','h','t'),
    Edit('ui','h','u'),Edit('vi','h','v'),Edit('wi','h','w'),Edit('xi','h','x'),Edit('yi','h','y'),Edit('zi','h','z'),Edit('ha','i','a'),Edit('hb','i','b'),Edit('hc','i','c'),Edit('hd','i','d'),Edit('he','i','e'),Edit('hf','i','f'),Edit('hg','i','g'),Edit('hh','i','h'),Edit('hj','i','j'),
    Edit('hk','i','k'),Edit('hl','i','l'),Edit('hm','i','m'),Edit('hn','i','n'),Edit('ho','i','o'),Edit('hp','i','p'),Edit('hq','i','q'),Edit('hr','i','r'),Edit('hs','i','s'),Edit('ht','i','t'),
    Edit('hu','i','u'),Edit('hv','i','v'),Edit('hw','i','w'),Edit('hx','i','x'),Edit('hy','i','y'),Edit('hz','i','z')
  ])

  print "***Code Sanity Check***"  
  print "\n\nDelete edits for 'hi'"
  checkOverlap(set(editModel.deleteEdits('hi')), DELETE_EDITS)
  print "\n\nInsert edits for 'hi'"
  checkOverlap(set(editModel.insertEdits('hi')), INSERT_EDITS)
  print "\n\nTranspose edits for 'hi'"
  checkOverlap(set(editModel.transposeEdits('hi')), TRANPOSE_EDITS)
  print "\n\nReplace edits for 'hi'"
  checkOverlap(set(editModel.replaceEdits('hi')), REPLACE_EDITS)

if __name__ == "__main__":
  main()
