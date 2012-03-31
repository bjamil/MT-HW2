#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple


# load in command line options
optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=30, type="int", help="Limit on number of translations to consider per phrase (default=30)")
optparser.add_option("-s", "--stack-size", dest="s", default=150, type="int", help="Maximum stack size (default=150)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]


# load translation and language models
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)

# load french sentences that need decoding
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

# define our hypothesis tuples
hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage, start, end")

# define variables
d = 3 # distortion
verbose = False


def expand_hypothesis(cost, h, f, start, end):
    """ expand the hypothesis with the phrase in the french sentence b/w start and end """
    
    # make sure we can translate the given phrase:
    
    if f[start:end] not in tm:
        return None # phrase not in tm, we can't translate it

    for i in range(start, end):
        if h.coverage[i] is 1:
            return None     # already translated a portion of this phrase; can't retranslate


    # we can translate it.
    
    # compute the new coverage vector
    cover = [x for x in h.coverage]
    for i in range(start, end):
        cover[i] = 1
    cover = tuple(cover)

    # expand the hypothesis with each possible phrase translation
    new_hypothesises = []

    for phrase in tm[f[start:end]]:
        
        # compute the new log probability of the new hypothesis
        logprob = h.logprob + phrase.logprob

        # get new language model score
        lm_state = h.lm_state
        for word in phrase.english.split():
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob

        # add in the probability that this is the end of the sentence, if applicable
        logprob += lm.end(lm_state) if sum(cover)==len(f) else 0.0

        # add in the future cost estimate
        logprob += estimate_future_cost(cost, h, f)

        # create a new hypothesis with this phrase translation
        new_hypothesis = hypothesis(logprob, lm_state, h, phrase, cover, start, end)
        new_hypothesises.append(new_hypothesis)
    
    # return all new hypothesises
    return new_hypothesises

def compute_max_cost(f):
  """ returns the max cost of this phrase, if it exists in the translation table;
      else returns -infinite """
  # initialize max cost to an arbitrarily small number  
  max_cost = float('-inf')

  if f in tm:
    # if this phrase exists in our translation table, find its most probable translation
    # (i.e. the one with the highest cost) 
    for phrase in tm[f]:
      english = phrase.english.split()
      cost = phrase.logprob   # translation probility
      lm_state = (english[0], )

      for word in english[1:]:   # language model probability
        (lm_state, word_logprob) = lm.score(lm_state, word)
        cost += word_logprob

      if cost > max_cost:
        max_cost = cost
        
  return max_cost


        
def construct_lookahead_table(f):
  """ constructs a future cost lookup table for the given french sentence """

  cost = {} # our lookup table

  # build the table incrementally ...1 word, then 2 words, then 3, and so on
  for length in range(1, len(f)+1):
    for start in range(len(f)+1-length):
      end = start+length

      # set the initial cost to an arbitrarily small number (we wanna maximize cost)
      # if it does not exist in our translation table. Else, set it to its largest
      # phrase translation model + language model probability
      cost[(start, end)] = compute_max_cost(f[start:end])


      # set the final cost for this start/end sequence as the longest cost path
      # among its building blocks OR the cost of its phrasal translation, whichever
      # is larger
      for i in range(start, end-1):
        if cost[(start, i+1)] + cost[(i+1, end)] > cost[(start, end)]:
          cost[(start, end)] = cost[(start, i+1)] + cost[(i+1, end)]
  
  return cost


def estimate_future_cost(cost, h, f):
  """ estimates the future cost of the given hypothesis using the provided lookahead table (cost) """

  fc = 0.0

  # the total future cost is the sum of the future costs of each untranslated spans

  # find each untranslated span and add its future cost to our total:
  i = 0
  start = 0
  end = 0
  gap = False
  while i < len(f):
    if h.coverage[i] is 0:
      if not gap:
        gap = True
        start = i
    else:
      if gap:
        end = i
        gap = False
        fc += cost[(start, end)]
    i+=1
  if gap:
    # the last untranslated span reaches the end of our coverage vector
    # add in the last span's cost too 
    fc += cost[(start, len(f))]

  return fc

  
  

# begin decoding
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    # set up all the stacks for this sentence (including the empty stack)
    stacks = [{} for _ in f] + [{}]

    # create an initial hypothesis and add it to the stack
    cover = tuple([0 for _ in f])
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, cover, 0, 0)

    stacks[0][((0,0), cover, lm.begin())] = initial_hypothesis # each hypothesis is unique identified by the
                                                    # the last phrase it translated and
                                                    # its coverage. it can recombine with other
                                                    # similar hypothesises

    # compute lookahead table
    cost = construct_lookahead_table(f)

    # expand the top k hypothesises in each stack (except the last one since we can't expand
    # it any further anyway)
    for (x, stack) in enumerate(stacks[:-1]):
        #prune the stack to keep just the top k hypothesises in it only
        for h in sorted(stack.itervalues(), key = lambda h: -h.logprob)[:opts.s]:
            if verbose:
                print "we have ", len(stack), "hypthesises in stack" , x
            # append phrases that came before the phrase in the current hypoth
            for i in range(0, h.start):
                for j in range(i+1, h.start+1):
                    new_hypothesises = expand_hypothesis(cost, h, f, i, j)

                    if new_hypothesises: # if it isn't None

                        #add each new hypothesis to the appropriate stack:
                        for new_hypothesis in new_hypothesises:
                            new_key = ((new_hypothesis.start, new_hypothesis.end), new_hypothesis.coverage, new_hypothesis.lm_state)
                            new_stack_num = sum(new_hypothesis.coverage)

                            # if we haven't seen this hypothesis before OR if we have but its prev score was less than its new score,
                            # add in this new hypothesis to the appropriate stack
                            if new_key not in stacks[new_stack_num] or stacks[new_stack_num][new_key].logprob < new_hypothesis.logprob:
                                stacks[new_stack_num][new_key] = new_hypothesis
            
            # append phrases that came after the phrase in the current hypothesis
            for i in range(h.end, len(f)):
                for j in range(i+1, len(f)+1):
                    new_hypothesises = expand_hypothesis(cost, h, f, i, j)
                    
                    if new_hypothesises: # if it isn't None
                                                
                         # add each new hypothesis to the appropriate stack
                         for new_hypothesis in new_hypothesises:
                            new_key = ((new_hypothesis.start, new_hypothesis.end), new_hypothesis.coverage, new_hypothesis.lm_state)
                            new_stack_num = sum(new_hypothesis.coverage)

                            # if we haven't seen this hypothesis before OR if we have but its prev score was less than its new score,
                            # add in this new hypothesis to the appropriate stack
                            if new_key not in stacks[new_stack_num] or stacks[new_stack_num][new_key].logprob < new_hypothesis.logprob:
                                stacks[new_stack_num][new_key] = new_hypothesis

    # find the best hypothesis for this sentence
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    def extract_english(h): 
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)

