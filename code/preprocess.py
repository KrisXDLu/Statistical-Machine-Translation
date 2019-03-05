import re

SEPARATE_1001299944 = ",:;()+—<>=\""
def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function
    out_sentence = "SENTSTART"
    in_paren = False
    out_sentence = in_sentence.lower()
    if out_sentence[-1] in ".?!":
    	out_sentence = out_sentence[:-1] + " " + out_sentence[-1]

    for punc in SEPARATE_1001299944:
    	out_sentence = out_sentence.replace(punc, " " + punc + " ")

    out_sentence = re.sub(r"\s{2,}", " ", out_sentence)

    if language == 'f':
    	out_sentence = re.sub(r"(\s[cdjlmnst]')([a-z])", r"\1 \2", out_sentence)
    	out_sentence = re.sub(r"(qu')([a-z])", r"\1 \2", out_sentence)
    	out_sentence = re.sub(r"([a-z]')(on|il)\s", r"\1 \2 ", out_sentence)
    	out_sentence = re.sub(r"(d')( )(abord|accord|ailleurs|habitude)", r"\1\3", out_sentence)

    out_sentence += " SENTEND"
    out_sentence = re.sub(r"\s{2,}", " ", out_sentence)

    return out_sentence