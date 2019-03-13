import math

def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.
    
    INPUTS:
    sentence :  (string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
    references: (list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
    n :         (int) one of 1,2,3. N-Gram level.

    
    OUTPUT:
    bleu_score :    (float) The BLEU score
    """
    
    #TODO: Implement by student.
    candidate_list = candidate.strip().split()
    c_reference = ' ' + '  '.join(references) + ' '
    p2, p3 = 1, 1
    C = 0
    N = 0
    for i in candidate_list:
        c_i = ' '+ i + ' '
        if c_i in c_reference:
            C += 1
        N += 1
    p1 = C / N
    bleu_score = p1
    if n >= 2:
        C2 = 0
        for i in range(len(candidate_list)-1):
            c_i2 = ' ' + candidate_list[i] + ' ' + candidate_list[i + 1] + ' '
            if c_i2 in c_reference:
                C2 += 1
        p2 = C2/ (N-1)
        bleu_score = p2
    if n == 3:
        C3 = 0
        for i in range(len(candidate_list)-2):
            c_i3 = ' ' + candidate_list[i] + ' ' + candidate_list[i + 1] + ' ' + candidate_list[i + 2] + ' '
            if c_i3 in c_reference:
                C3 += 1
        p3 = C3/ (N-2)
        bleu_score = p3
    if brevity:
        c_len = len(candidate_list)
        r_len = [len(r.split()) for r in references]
        s_distance = float('inf')
        nearest_r = 0
        for r in r_len:
            if abs(c_len - r) < s_distance:
                nearest_r = r
                s_distance = abs(c_len - r)
        brevity_num = nearest_r/c_len
        BP = 1
        if brevity_num >= 1:
            BP = math.exp(1-brevity_num)
        bleu_score = BP * (p1 * p2 * p3)**(1/n)
    return bleu_score

candidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
candidate2 = "It is to insure the troops forever hearing the activity guidebook that party direct"
# candidate3 = "I fear David"
# references_list2 = ['I am afraid Dave', 'I am scared Dave', 'I have fear David']
references_list = ["It is a guide to action that ensures that the military will forever heed Party commands",
"It is the guiding principle which guarantees the military forces always being under command of the Party",
"It is the practical guide for the army always to heed the directions of the party"]
print(BLEU_score(candidate2, references_list, 1, True))
print(BLEU_score(candidate2, references_list, 2, True))
print(BLEU_score(candidate2, references_list, 3, True))
print(BLEU_score(candidate2, references_list, 1))
print(BLEU_score(candidate2, references_list, 2))
print(BLEU_score(candidate2, references_list, 3))
