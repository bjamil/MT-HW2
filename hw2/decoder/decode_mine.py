#!/usr/bin/env python
###############################################################################################################
# decode_mine.py                                                                                              #
# Author      : Beenish Jamil                                                                                 #
# Assignment  : CS468 - HW 2                                                                                  #
# Description : An implementation of the stack decoding method for phrase based SMT.                          #
#               Note that this model allows all permutations of the phrases with no reordering limits.        #
#               This implementation also does not include any future cost estimation heuristics.              #
###############################################################################################################
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
optparser.add_option("-s", "--stack-size", dest="s", default=250, type="int", help="Maximum stack size (default=250)")
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


def expand_hypothesis(h, f, start, end):
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

        # create a new hypothesis with this phrase translation
        new_hypothesis = hypothesis(logprob, lm_state, h, phrase, cover, start, end)
        new_hypothesises.append(new_hypothesis)
    
    # return all new hypothesises
    return new_hypothesises

        
    


# begin decoding
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    # set up all the stacks for this sentence (including the empty stack)
    stacks = [{} for _ in f] + [{}]

    # create an initial hypothesis and add it to the stack
    cover = tuple([0 for _ in f])
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, cover, 0, 0)

    stacks[0][((0,0), cover)] = initial_hypothesis  # each hypothesis is uniquely identified by the
                                                    # the last phrase it translated and
                                                    # its coverage. it can recombine with other
                                                    # hypothesises with the same properties

    # expand the top k hypothesises in each stack (except the last one since we can't expand
    # it any further anyway)
    for (x, stack) in enumerate(stacks[:-1]):
        #prune the stack to keep just the top k hypothesises in it only
        for h in sorted(stack.itervalues(), key = lambda h: -h.logprob)[:opts.s]:
            # append phrases that came before the phrase in the current hypoth
            for i in range(0, h.start):
                for j in range(i+1, h.start+1):
                    new_hypothesises = expand_hypothesis(h, f, i, j)

                    if new_hypothesises: # if it isn't None

                        #add each new hypothesis to the appropriate stack:
                        for new_hypothesis in new_hypothesises:
                            new_key = ((new_hypothesis.start, new_hypothesis.end), new_hypothesis.coverage)
                            new_stack_num = sum(new_hypothesis.coverage)

                            # if we haven't seen this hypothesis before OR if we have but its prev score was less than its new score,
                            # add in this new hypothesis to the appropriate stack
                            if new_key not in stacks[new_stack_num] or stacks[new_stack_num][new_key].logprob < new_hypothesis.logprob:
                                stacks[new_stack_num][new_key] = new_hypothesis
            
            # append phrases that came after the phrase in the current hypothesis
            for i in range(h.end, len(f)):
                for j in range(i+1, len(f)+1):
                    new_hypothesises = expand_hypothesis(h, f, i, j)
                    
                    if new_hypothesises: # if it isn't None
                                                
                         # add each new hypothesis to the appropriate stack
                         for new_hypothesis in new_hypothesises:
                            new_key = ((new_hypothesis.start, new_hypothesis.end), new_hypothesis.coverage)
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

