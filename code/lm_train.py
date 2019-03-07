from preprocess import *
import pickle
import os
# from os.path import isfile, join

def lm_train(data_dir, language, fn_LM):
    """
    This function reads data from data_dir, computes unigram and bigram counts,
    and writes the result to fn_LM
    
    INPUTS:
    
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    
    OUTPUT
    
    LM          : (dictionary) a specialized language model
    
    The file fn_LM must contain the data structured called "LM", which is a dictionary
    having two fields: 'uni' and 'bi', each of which holds sub-structures which 
    incorporate unigram or bigram counts
    
    e.g., LM['uni']['word'] = 5         # The word 'word' appears 5 times
          LM['bi']['word']['bird'] = 2  # The bigram 'word bird' appears 2 times.
    """
    
    # TODO: Implement Function
    LM = {}
    LM["uni"] = {}
    LM["bi"] = {}
    # mypath = "../data/Hansard/Training/"
    
    onlyfiles = [f for f in os.listdir(data_dir) if os.isfile(os.join(data_dir, f))]
    for name in onlyfiles:
        with open(name, "r") as file:
            for line in file.readlines():
                sentence = preprocess(line).split(" ")
                for i in range(len(sentence)):
                    word = sentence[i]
                    if len(word.strip()) > 0:
                        if word in LM["uni"]:
                            LM["uni"][word] += 1
                        else:
                            LM["uni"][word] = 1
                        if i > 0:
                            item = sentence[i-1]
                            if item in LM["bi"]:
                                if word in LM["bi"][item]:
                                    LM["bi"][item][word] += 1
                                else:
                                    LM["bi"][item][word] = 1
                            else:
                                LM["bi"][item] = {}
                                LM["bi"][item][word] = 1
    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(LM, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return LM