from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing
    
    INPUTS:
    sentence :  (string) The PROCESSED sentence whose probability we wish to compute
    LM :        (dictionary) The LM structure (not the filename)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta :     (float) smoothing parameter where 0<delta<=1
    vocabSize : (int) the number of words in the vocabulary
    
    OUTPUT:
    log_prob :  (float) log probability of sentence
    """
    
    #TODO: Implement by student.
    if smoothing:
        # delta smooth estimate
        if delta == 0 or vocabSize == 0:
            print("delta or vocabSize not specified for smoothing")
            return 0
    else:
        delta = 0
        vocabSize = 0

    log_prob = 0
    bi_log_prob = float("-inf")

    words = sentence.split(" ")
    for i in range(1, len(words)):
        cur = words[i]
        pre = words[i-1]
        num_cur = 0
        num_pre = 0
        if pre in LM["uni"]:
            num_pre = LM["uni"][pre]
            if cur in LM["bi"][pre]:
                num_cur = LM["bi"][pre][cur]
        if smoothing:
            prob = (num_cur + delta)/(num_pre + (delta * vocabSize))
            bi_log_prob = log(prob, 2)
        elif not smoothing and num_cur != 0:
            bi_log_prob = log(num_cur /num_pre, 2)
        # max likelihood esti
        log_prob += bi_log_prob

    return log_prob



