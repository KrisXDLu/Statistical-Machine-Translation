from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
    Implements the training of IBM-1 word alignment algoirthm. 
    We assume that we are implemented P(foreign|english)
    
    INPUTS:
    train_dir :     (string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    max_iter :      (int) the maximum number of iterations of the EM algorithm
    fn_AM :         (string) the location to save the alignment model
    
    OUTPUT:
    AM :            (dictionary) alignment model structure
    
    The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
    is the computed expectation that the foreign_word is produced by english_word.
    
            LM['house']['maison'] = 0.5
    """
    AM = {}
    
    # Read training data
    eng, fre = read_hansard(train_dir, num_sentences)
    
    # Initialize AM uniformly
    AM = initialize(eng, fre)
    
    # Iterate between E and M steps

    for i in range(max_iter):
        AM = em_step(AM, eng, fre)
        print(i)

    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.
    
    INPUTS:
    train_dir :     (string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    
    
    Make sure to preprocess!
    Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
    
    Make sure to read the files in an aligned manner.
    """
    # TODO
    files = os.listdir(train_dir)
    files = set([f[:-1] for f in files])
    # vocab_size = len(LM["uni"])
    english = []
    french = []
    count = 0
    for ffile in files:
        eng_file = open(train_dir+ffile + "e", "r")
        fre_file = open(train_dir+ffile + "f", "r")
        eng_lines = eng_file.readlines()
        fre_lines = fre_file.readlines()
        for i in range(len(eng_lines)):
            if count == num_sentences:
                return english, french
            count += 1
            english.append(preprocess(eng_lines[i].strip(), "e").strip().split(" ")[1:-1])
            french.append(preprocess(fre_lines[i].strip(), "f").strip().split(" ")[1:-1])
    return english, french

def initialize(eng, fre):
    """
    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """
    # TODO
    alignment = {}
    AM = {}

    for i in range(len(eng)):
        sent_eng = eng[i]
        for word in sent_eng:
            if not word in alignment:
                alignment[word] = []
            update = [item for item in fre[i] if item not in alignment[word]]
            alignment[word] += update
    for key in alignment:
        AM[key] = {}
        words = alignment[key]
        prob = 1/float(len(words))
        for item in words:
            AM[key][item] = prob

    AM["SENTSTART"] = {"SENTSTART": 1}
    AM["SENTEND"] = {"SENTEND": 1}

    return AM
    
def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    # TODO  
    # set tcount(f, e) 
    tcount = {}
    #  set total(e) 
    total = {}
    for word in t:
        total[word] = 0
        tcount[word] = {}
        for french in t[word]:
            tcount[word][french] = 0
    #  for each sentence pair (F, E) in training corpus:
    for i in range(len(eng)):
        count_E = eng[i]
        E = set(count_E)
        count_F = fre[i]
        F = set(count_F)
        #  for each unique word f in F:
        for f in F:
            denom = 0
            for e in E:
                denom += t[e][f] * count_F.count(f)
            for e in E:
                num = t[e][f] * count_F.count(f) * count_E.count(e) / float(denom)
                tcount[e][f] += num
                total[e] += num
    for e in total:
        for f in tcount[e]:
            if not e in ["SENTSTART", "SENTEND"]:
                t[e][f] = tcount[e][f] / total[e]
             
    return t


# t = initialize(['the blue cat blue'.split(' '), 'the red dog'.split(' ')],['le chat bleu'.split(' '), 'le chein rouge'.split(' ')])
# print(t)
# print(em_step(t, ['the blue cat'.split(' '), 'the red dog'.split(' ')],['le chat bleu'.split(' '), 'le chein rouge'.split(' ')]))
r = align_ibm1('../data/Hansard/Training/', 1000, 100, "am_temp")
print(r['the']['le'])
print(r['the']["l'"])
print(r['the']['la'])
