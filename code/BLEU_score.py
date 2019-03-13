import math

def BLEU_score(candidate, references, n, brevity=False):
        """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.
    
    INPUTS:
        sentence :(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
        references:(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
n :(int) one of 1,2,3. N-Gram level.

        
        OUTPUT:
        bleu_score :(float) The BLEU score
        """

        #TODO: Implement by student.
        bleu_score = 0
        score = 0
        sentence = candidate.split()

        # print(len(sentence[1:1+n]) == n)
        denom = len(sentence)
        if not brevity:
            for i in range(len(sentence)-n+1):
                phrase = " ".join(sentence[i:i+n])
                phrase = " " + phrase + " "
                # print(phrase)
                # print(len(phrase))
                for sent in references:
                    sent = " " + sent + " "
                    if phrase in sent:
                        score += 1
                        break

            # print(score)
            # print(denom)
            bleu_score = score/(denom-n+1)
            return bleu_score
        else:

            diff = float('inf')
            sim = -1
            for i in range(len(references)):
                cur = references[i].split()
                size = len(cur)
                # print(size)
                if abs(size - denom) < diff:
                    diff = abs(size - denom)
                    sim = i
            # print(sim)
            # print(diff)
            brevity = len(references[sim].split())/denom
            # print("final")
            # print(len(references[sim]))

            if brevity < 1:
                bp = 1
            else:
                bp = math.exp(1-brevity)
            bleu_score = 1
            # print("denom " + str(denom-i+1))
            for i in range(1,n+1):
                score = 0
                for j in range(len(sentence)-i+1):
                    phrase = " ".join(sentence[j:j+i])
                    phrase = " " + phrase + " "
                    for sent in references:
                        sent = " " + sent + " "
                        if phrase in sent:
                            score += 1
                            break
                bleu_score = bleu_score * score/(denom-i+1)
            bleu_score = bleu_score ** (1/n)
            bleu_score = bp * bleu_score
            return bleu_score






# candidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
# candidate2 = "It is to insure the troops forever hearing the activity guidebook that party direct"
# candidate3 = "I fear David"
# # references_list2 = ['I am afraid Dave', 'I am scared Dave', 'I have fear David']
# references_list = ["It is a guide to action that ensures that the military will forever heed Party commands",
# "It is the guiding principle which guarantees the military forces always being under command of the Party",
# "It is the practical guide for the army always to heed the directions of the party"]
# print(BLEU_score(candidate2, references_list, 1, True))
# print(BLEU_score(candidate2, references_list, 2, True))
# print(BLEU_score(candidate2, references_list, 3, True))
# print(BLEU_score(candidate2, references_list, 1))
# print(BLEU_score(candidate2, references_list, 2))
# print(BLEU_score(candidate2, references_list, 3))



















