# Shared library for all submit scripts.
# Each assignment should create a subclass that inherits
# from Submitter and implements the required assignment-specific
# methods.

import urllib
import urllib2
import hashlib
import random
import email
import email.message
import email.encoders
import StringIO
import sys

#URL TO CHANGE EVERY YEAR: portion of URL after https://stanford.coursera.org/
#ie for winter 2014 cs124-002 and winter 2015 cs124-003
BASE_CLASS_URL = "cs124-003"

#boolean for submission testing purposes when assignment hasn't been released to class yet
RELEASED_TO_CLASS = True

# Subclasses must implement the following methods:
# validParts()
# sources()
# homework_id()
# output()
class Submitter(object):
  def submit(self, partId):
    print '==\n== [nlp] Submitting Solutions | Programming Exercise %s\n=='% self.homework_id()
    if(not partId):
      partId = self.promptPart()

    partNames = self.validParts()
    if not self.isValidPartId(partId):
      print '!! Invalid homework part selected.'
      print '!! Expected an integer from 1 to %d.' % (len(partNames) + 1)
      print '!! Submission Cancelled'
      return

    (login, password) = self.loginPrompt()
    if not login:
      print '!! Submission Cancelled'
      return

    print '\n== Connecting to coursera ... '

    # Setup submit list
    if partId == len(partNames) + 1:
      submitParts = range(1, len(partNames) + 1)
    else:
      submitParts = [partId]

    for partId in submitParts:
      # Get Challenge
      (login, ch, state, ch_aux) = self.getChallenge(login, partId)
      if((not login) or (not ch) or (not state)):
        # Some error occured, error string in first return element.
        print '\n!! Error: %s\n' % login
        return

      # Attempt Submission with Challenge
      ch_resp = self.challengeResponse(login, password, ch)
      (result, string) = self.submitSolution(login, ch_resp, partId, self.output(partId, ch_aux), \
                                      self.source(partId), state, ch_aux)
      print '== [nlp] Submitted Homework %s - Part %d - %s' % \
            (self.homework_id(), partId, partNames[partId - 1])
      print '== %s' % string.strip()
      if (string.strip() == 'Exception: We could not verify your username / password, please try again. (Note that your password is case-sensitive.)'):
        print '== The password is not your login, but a 10 character alphanumeric string displayed on the top of the Assignments page'


  def promptPart(self):
    """Prompt the user for which part to submit."""
    print('== Select which part(s) to submit: ' + self.homework_id())
    partNames = self.validParts()
    srcFiles = self.sources()
    for i in range(1,len(partNames)+1):
      print '==   %d) %s [ %s ]' % (i, partNames[i - 1], srcFiles[i - 1])
    print '==   %d) All of the above \n==\nEnter your choice [1-%d]: ' % \
            (len(partNames) + 1, len(partNames) + 1)
    selPart = raw_input('')
    partId = int(selPart)
    if not self.isValidPartId(partId):
      partId = -1
    return partId


  def isValidPartId(self, partId):
    """Returns true if partId references a valid part."""
    partNames = self.validParts()
    return (partId and (partId >= 1) and (partId <= len(partNames) + 1))


  # =========================== LOGIN HELPERS ===========================

  def loginPrompt(self):
    """Prompt the user for login credentials. Returns a tuple (login, password)."""
    (login, password) = self.basicPrompt()
    return login, password


  def basicPrompt(self):
    """Prompt the user for login credentials. Returns a tuple (login, password)."""
    login = raw_input('Login (Email address): ')
    password = raw_input('Password: ')
    return login, password


  def getChallenge(self, email, partId):
    """Gets the challenge salt from the server. Returns (email,ch,state,ch_aux)."""
    url = self.challenge_url()
    if (RELEASED_TO_CLASS):
      values = {'email_address' : email, 'assignment_part_sid' : "%s-%s" % (self.homework_id(), partId), 'response_encoding' : 'delim'}
    else:
      values = {'email_address' : email, 'assignment_part_sid' : "%s-%s" % (self.homework_id(), partId), 'response_encoding' : 'delim'}
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    text = response.read().strip()

    # text is of the form email|ch|signature
    splits = text.split('|')
    if(len(splits) != 9):
      print 'Badly formatted challenge response: %s' % text
      return None
    return (splits[2], splits[4], splits[6], splits[8])



  def challengeResponse(self, email, passwd, challenge):
    sha1 = hashlib.sha1()
    sha1.update("".join([challenge, passwd])) # hash the first elements
    digest = sha1.hexdigest()
    strAnswer = ''
    for i in range(0, len(digest)):
      strAnswer = strAnswer + digest[i]
    return strAnswer


  def challenge_url(self):
    """Returns the challenge url."""
    return "https://stanford.coursera.org/" + BASE_CLASS_URL + "/assignment/challenge"

  def submit_url(self):
    """Returns the submission url."""
    return "https://stanford.coursera.org/" + BASE_CLASS_URL + "/assignment/submit"

  def submitSolution(self, email_address, ch_resp, part, output, source, state, ch_aux):
    """Submits a solution to the server. Returns (result, string)."""
    source_64_msg = email.message.Message()
    source_64_msg.set_payload(source)
    email.encoders.encode_base64(source_64_msg)

    output_64_msg = email.message.Message()
    output_64_msg.set_payload(output)
    email.encoders.encode_base64(output_64_msg)
    if (RELEASED_TO_CLASS):
      values = { 'assignment_part_sid' : ("%s-%s" % (self.homework_id(), part)), \
               'email_address' : email_address, \
               'submission' : output_64_msg.get_payload(), \
               'submission_aux' : source_64_msg.get_payload(), \
               'challenge_response' : ch_resp, \
               'state' : state \
               }
    else:
      values = { 'assignment_part_sid' : ("%s-%s-dev" % (self.homework_id(), part)), \
               'email_address' : email_address, \
               'submission' : output_64_msg.get_payload(), \
               'submission_aux' : source_64_msg.get_payload(), \
               'challenge_response' : ch_resp, \
               'state' : state \
               }
    url = self.submit_url()
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    string = response.read().strip()
    # TODO parse string for success / failure
    result = 0
    return result, string

  def source(self, partId):
    """Reads in the source files for a given partId."""
    src = ''
    src_files = self.sources()
    if partId <= len(src_files):
      flist = src_files[partId - 1]
      for fname in flist:
        # open the file, get all lines
        f = open(fname)
        src = src + f.read()
        f.close()
        src = src + '||||||||'
    return src
