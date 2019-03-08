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
    (train_e, train_f) = read_hansard(train_dir, num_sentences)
    # print(train_e)

    # Initialize AM uniformly
    AM = initialize(train_e, train_f)
    
    # Iterate between E and M steps
    for i in range(max_iter):
        AM = em_step(AM, train_e, train_f)
        print(i)

    # Save AM as file
    with open(fn_AM + '.pickle', 'wb') as file:
        pickle.dump(AM, file, protocol=pickle.HIGHEST_PROTOCOL)

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
    eng_line = []
    fre_line = []
    num_line = 0
    readfile = []
    for root, dirs, files in os.walk(train_dir):
        # files = sorted(files)
        for name in files:
            if name not in readfile and name[-1] in ['e', 'f']:
                e_file = name[:-1]+'e'
                f_file = name[:-1]+'f'
                e_path = os.path.join(root, e_file)
                f_path = os.path.join(root, f_file)
                readfile.append(e_file)
                readfile.append(f_file)
                if e_file in files and f_file in files:
                    print(e_path, f_path)
                    with open(e_path) as e_f:
                        e_content = e_f.readlines()
                    with open(f_path) as f_f:
                        f_content = f_f.readlines()
                    for line in range(len(e_content)):
                        if num_line == num_sentences:
                            return eng_line, fre_line
                        e_list = preprocess(e_content[line], 'e').split(' ')
                        f_list = preprocess(f_content[line], 'f').split(' ')
                        eng_line.append(e_list)
                        fre_line.append(f_list)
                        num_line += 1
    return (eng_line, fre_line)

def initialize(eng, fre):
    """
    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """
    AM = {'SENTSTART':{'SENTSTART':1}, 'SENTEND':{'SENTEND':1}}
    IM = {}
    for s in range(len(eng)):
        e_wordlist = eng[s][1:-1]
        f_wordlist = fre[s][1:-1]
        for e_word in e_wordlist:
            if e_word not in IM:
                IM[e_word] = []
            for f_word in f_wordlist:
                if f_word not in IM[e_word]:
                    IM[e_word].append(f_word)
    for e_IM in IM:
        AM[e_IM] = {}
        for f_IM in IM[e_IM]:
            AM[e_IM][f_IM] = 1/len(IM[e_IM])
    return AM
    
def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    # TODO
    # set tcount and total
    t_count = {}
    total = {}
    for e_word in t:
        total[e_word] = 0
        t_count[e_word] = {}
        for f_word in t[e_word]:
            t_count[e_word][f_word] = 0

    for i in range(len(fre)):
        e_sent = eng[i][1:-1]
        f_sent = fre[i][1:-1]
        e_uniq = set(e_sent)
        f_uniq = set(f_sent)
        for f in f_uniq:
            denom_c = 0
            for e in e_uniq:
                denom_c += t[e][f] * f_sent.count(f)
            for e in e_uniq:
                addon = t[e][f] * f_sent.count(f) * e_sent.count(e) / denom_c
                t_count[e][f] += addon
                total[e] += addon

    for e in total:
        for f in t_count[e]:
            if e not in ['SENTSTART', 'SENTEND']:
                t[e][f] = t_count[e][f]/total[e]
    return t

# t = initialize(['SENTSTART the blue cat blue SENTEND'.split(' '), 'SENTSTART the red dog SENTEND'.split(' ')],['SENTSTART le chat bleu SENTEND'.split(' '), 'SENTSTART le chein rouge SENTEND'.split(' ')])
# em_step(t, ['SENTSTART the blue cat SENTEND'.split(' '), 'SENTSTART the red dog SENTEND'.split(' ')],['SENTSTART le chat bleu SENTEND'.split(' '), 'SENTSTART le chein rouge SENTEND'.split(' ')])
r = align_ibm1('../data/Hansard/Training/', 1000, 100, "am_temp")
print(r['the']['le'])
print(r['the']["l'"])
print(r['the']['la'])