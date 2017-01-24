# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:16:40 2016

@author: hazie
"""
import re

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''
word = "ahead"

print("Cleaning text...")
only_letters = re.sub("[^a-zA-Z]"," ",sample_memo)
print("Cleaning text completed")
print("Splitting text...")
split_text = only_letters.split()
print("Splitting text completed")

def CalcProb(word_list):
    print("Calculating probability for each element for every keys...\n")
    total_count = 0
    for i,word in enumerate(word_list):
        total_count += word_list[word]

    for j,wrd in enumerate(word_list):
        word_list[wrd] = round(word_list[wrd]/total_count,3)
    print(word_list)
    print("\n")
    return word_list

def Count(word_list,word):
    from collections import Counter
    cnt = Counter()
    for i, Word in enumerate(split_text):
        if Word == word:
            cnt[word_list[i+1]]+=1
    
    new_word_list = dict(cnt)
    return new_word_list
    
new_word_list = CalcProb(Count(split_text,word))
new_word_list_2 = []
for i, wd in enumerate(new_word_list):
    new_word_list_2.append(CalcProb(Count(split_text,wd)))
highest_probability = 0
current_probability = 0
word_interest = []

for j,w in enumerate(new_word_list):
    for k,W in enumerate(new_word_list_2):
        temp_list = new_word_list_2[j]
        for l,Ww in enumerate(temp_list):
            current_probability = temp_list[Ww]*new_word_list[w]
            print("1st word:{}, 2nd word:{}, probability:{:.3f}\n".format(w,Ww,current_probability))
            if current_probability > highest_probability:
                highest_probability = current_probability
                word_interest = Ww
                
    print("{} has highest probability: {:3f}".format(word_interest,highest_probability))