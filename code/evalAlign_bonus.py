#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

from decode_bonus import *
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

Across all the data, we see no difference in changing and manipulating the
N, the maximum number of translations for each word in the sentence or the
MAXTRANS, the maximum number of greedy transformations we perform.  The only
change we observed are bleu score increasing as we increase the training size
like we discussed in task 5.
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
    with open(indir+"Testing/Task5.f", "r") as f:
        lines = f.readlines()
    
    LM = _getLM(indir + "Training/", "f", "./lm", False)

    eng_ref = open(indir+"Testing/Task5.e", "r").readlines()
    goo_ref = open(indir+"Testing/Task5.google.e", "r").readlines()
    eng_ref = [preprocess(i, 'e') for i in eng_ref]
    goo_ref = [preprocess(j, 'e') for j in goo_ref]
    
    f = open("TaskBonus.txt", 'w+')

    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    # eng_bleu_list = []
    # goo_bleu_list = []
    print("AM training")
    AMs = [_getAM(indir + "Training/", size, 20, "./am"+str(size), False) for size in alignment_size]
    m_list = [2, 10, 32, 64, 128, 256]

    for i in range(len(alignment_size)):
        size = alignment_size[i]
        # f.write(f"\n### Evaluating AM model: {AMs[i]} ### \n")
        f.write(f"\n### Evaluating AM model: AM{i} ### \n")
        f.write("Training size "+str(size)+":\n")
        AM = AMs[i]
        print(i)
        for j in range(1,6):
            for m in m_list:
                f.write(f"\n### Evaluating decode N={j}, maxtran={m} ### \n")
                decoded_eng = [decode_bonus(preprocess(line, "f"), LM, AM, j, m) for line in lines]
                    
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