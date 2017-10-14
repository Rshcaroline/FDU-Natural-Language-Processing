class Sentence:
  """Contains a list of Datums."""

  def __init__(self, sentence=[]):
    if(type(sentence) == type([])):
      self.data = list(sentence) 
    else:
      self.data = list(sentence.data)
  
  def getErrorSentence(self):
    """Returns a list of strings with the sentence containing all errors."""
    errorSentence = []
    for datum in self.data:
      if datum.hasError():
        errorSentence.append(datum.error)
      else:
        errorSentence.append(datum.word)
    return errorSentence

  def getCorrectSentence(self):
    """Returns a list of strings with the sentence containing all corrections."""
    correctSentence = []
    for datum in self.data:
      correctSentence.append(datum.word)
    return correctSentence

  # Checks if a list of strings is a correction of this sentence.
  def isCorrection(self, candidate):
    if len(self.data) != len(candidate):
      return False
    for i in range(0,len(self.data)):
      if not candidate[i] == self.data[i].word:
        return False
    return True

  def getErrorIndex(self):
    for i in range(0, len(self.data)):
      if self.data[i].hasError():
        return i
    return -1

  def len(self):
    return len(self.data)

  def get(self, i):
    return self.data[i]

  def put(self, i, val):
    self.data[i] = val

  def cleanSentence(self):
    """Returns a new sentence with all datum's having error removed."""
    sentence = Sentence()
    for datum in self.data:
      clean = datum.fixError()
      sentence.append(clean)
    return sentence

  def isEmpty(self):
    return len(self.data) == 0

  def append(self, item):    
    self.data.append(item)

  def __len__(self):
    return len(self.data)

  def __str__(self):
    str_list = []
    for datum in self.data:
      str_list.append(str(datum))
    return ' '.join(str_list)
