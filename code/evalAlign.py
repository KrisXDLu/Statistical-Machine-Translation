#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

from decode import *
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

Generally the bleu score increases with the training data size increase, 
it could be the result of the larger training size is more fit to the language.
Since our data size is still relatively small to the language size. Higher bleu score 
indicates better translation.  Larger data size makes our code understand the language better.
However, we didn't see a such significant difference when our tranining size changed from 15k
to 30k, this could because our model is too fit on the training data, and not as general anymore
on the testing data.

There is a lot of zeros in three gram possible because translating is much harder and our 
implementation is not as good as in terms of ordering and the correct translation.  
Also, three gram is harder to much since language is very complex and different ordering 
and difference choice of words in a sentence still could lead to the same meaning.

When I examin the two references, I find that they are very much similar to each other.
It is better to find more than one reference since there are many ways in translating 
the same language.
"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    LM = {}

    if use_cached:
        fname = fn_LM + ".pickle"
        if os.isfile(fname):
            file = open(fname, "rb")
            LM = pickle.load(file)
    else:
        LM = lm_train(data_dir, language, fn_LM)

    return LM


def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    AM = {}

    if use_cached:
        fname = fn_AM + ".pickle"
        if os.isfile(fname):
            file = open(fname, "rb")
            AM = pickle.load(file)
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)

    return AM

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    bleu = []
    for i in range(len(eng_decoded)):
        reference = [eng[i], google_refs[i]]
        score = BLEU_score(eng_decoded[i], reference, n, brevity=True)
        bleu.append(score)
    return bleu
    # for sent in eng_decoded:
    #     eng_bleu.append(BLEU_score(sent, eng, n, brevity=False))
    #     google_bleu.append(BLEU_score(sent, google_refs, n, brevity=False))
    # return eng_bleu, google_bleu


def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    decoded_eng = []
    # indir = "../data/Hansard/"
    indir = "/u/cs401/A2_SMT/data/Hansard/"
    alignment_size = [1000, 10000, 15000, 30000]
    # mydir = "/h/u4/g6/00/luxiaodi/csc401/a2/"
    with open(indir+"Testing/Task5.f", "r") as f:
        lines = f.readlines()
    
    LM = _getLM(indir + "Training/", "f", "./lm", False)

    eng_ref = open(indir+"Testing/Task5.e", "r").readlines()
    goo_ref = open(indir+"Testing/Task5.google.e", "r").readlines()
    eng_ref = [preprocess(i, 'e') for i in eng_ref]
    goo_ref = [preprocess(j, 'e') for j in goo_ref]
    
    f = open("Task5.txt", 'w+')

    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    # eng_bleu_list = []
    # goo_bleu_list = []
    print("AM training")
    AMs = [_getAM(indir + "Training/", size, 20, "./am"+str(size), False) for size in alignment_size]


    for i in range(len(alignment_size)):
        size = alignment_size[i]
        # f.write(f"\n### Evaluating AM model: {AMs[i]} ### \n")
        f.write(f"\n### Evaluating AM model: AM{i} ### \n")
        f.write("Training size "+str(size)+":\n")
        AM = AMs[i]
        print(i)
        decoded_eng = [decode(preprocess(line, "f"), LM, AM) for line in lines]
            
        for n in range(1, 4):
            bleu = _get_BLEU_scores(decoded_eng, eng_ref, goo_ref, n)
            f.write("BLEU_score with n:"+str(n)+"\n")
            f.write("bleu: "+str(bleu)+"\n\n")
            # f.write("Google bleu: "+str(google_bleu))
            # eng_bleu_list.append(eng_bleu)
            # goo_bleu_list.append(google_bleu)
        f.write("\n\n")
    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()
    # indir = "../data/Hansard/"
    # AM = _getAM(indir + "Training/", 1000, 20, indir + "../am"+str(1000), False)
    # LM = _getLM(indir + "Training/", "f", indir + "../lm", False)
    # line = "Dans le monde reel, il n'y a rien de mal a cela.\n"
    # print(decode(preprocess(line, "f"), LM, AM))
    main(args)