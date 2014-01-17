'''
Created on Oct 28, 2013

@author: samshuster
'''
from collections import defaultdict

'''Takes x,cur_y,prev_y,position'''

def atomic_tag_label(words, curr_tag, prev_tag, i, tag='NN', word='computer'):
    try:
        if curr_tag == tag and words[i].lower() == word.lower():
            return 1
    except:
        return 0
    return 0
    
def bigram_tag_label(words, curr_tag, prev_tag, i, tag='NN',tag2 = 'NN'):
    if curr_tag == tag and prev_tag == tag2:
        return 1
    else:
        return 0

def suffix(words,curr_tag,prev_tag,i,suff_length=3, suffix = 'ing', tag='NN'):
    w = words[i].lower()
    if w[-suff_length:] == suffix and curr_tag == tag:
        return 1
    else:
        return 0
    
def trial(words,curr_tag,prev_tag,i,tag,tag2):
    print tag, tag2
    if curr_tag == tag and prev_tag == tag2:
        return 1
    else:
        return 0

def get_feature_funcs():
    t = ['NN','DT','NNP','TO','IN','RB','VBG','VBN','VBD']
    for t1 in t:
        for t2 in t:
            func = lambda words,curr_tag,prev_tag,i,t1=t1,t2=t2: trial(words,curr_tag,prev_tag,i,t1,t2)
            print func(['hey','you','are','cool'],'NN','NNP',0)