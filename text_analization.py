import pandas as pd
import numpy as np
import math
import gensim
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import operator

print("loading model")
Vec = gensim.models.KeyedVectors.load_word2vec_format('.\GoogleNews-vectors-negative300.bin', binary=True)
print("loaded model!")
model = None
try:
    model = Vec.wv
except Exception:
    model = Vec
q_verbs = ['be', 'is', 'are', 'was', 'were', 'do', 'did', 'does', 'have', 'has', 'had', 'can', 'could', 'may', 'might', 'must', 'will', 'would', 'should']
pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'WDT', 'WP', 'WP$', 'WRB']

def is_english_word(word):
    global model
    if word in model:
        return True
    else:
        return False

def normalization(raw_string):
    exclude = string.punctuation
    new_string = ''.join(ch for ch in raw_string if ch not in exclude)
    new_string = new_string.lower()
    return new_string

def isfaq(sentence):
    words = sentence.split(" ")
    digit = 0
    words_nodig = []
    for w in words:
        if w.isdigit():
            digit += 1
        else:
            words_nodig.append(w)
    if digit > len(words)-2:
        return False
    else:
        engl = 0
        words_nodig_en = []
        for w in words_nodig:
            if is_english_word(w):
                engl += 1
                words_nodig_en.append(w)
        if engl < 2:
            return False
        else:
            global q_verbs, pos_tags
            flag = 0
            for w in words_nodig_en:          
                if w.lower() in q_verbs:
                    flag += 1
            poss = nltk.pos_tag(words_nodig)
            for pos in poss:
                if pos[1] in pos_tags:
                    flag += 1
            if flag > 1:
                return True
            else:
                return False

data = pd.read_csv("data.csv", encoding='latin-1')
faq_questions_raw = data["Q"]
faq_answers = data["A"]
faq_questions = []
for quest in faq_questions_raw:
    faq_questions.append(normalization(quest))

def cosine(vector1, vector2):
    ab = 0
    a = 0
    b = 0
    for i in range(vector1.shape[0]):
        ab += vector1[i]*vector2[i]
        a += math.pow(vector1[i],2)
        b += math.pow(vector2[i],2)
    return ab/(math.sqrt(a)*math.sqrt(b))

def sent_feat(sent, model):
    words = sent.split(" ")
    temp = []
    for w in words:
        if w in model:
            temp.append(model[w])
    if temp:
        temp = list((np.array(temp)).mean(0)) + list(np.percentile(np.array(temp), 25, axis=0))
    else:
        temp = list(np.zeros(300))
    return np.array(temp)

def find_answer(user_qestion):
    user_qestion = normalization(user_qestion)
    if not isfaq(user_qestion):
        return "This is not FAQ!"
    else:
        global faq_questions
        global faq_answers
        answer_dict = {}
        global model
        quest_vector = sent_feat(user_qestion, model)
        better_syn = ''
        max_cos = -1
        for syn in faq_questions:
            syn_vector = sent_feat(syn, model)
            cos = cosine(quest_vector, syn_vector)
            if cos > max_cos:
                max_cos = cos
                better_syn = syn
        index = faq_questions.index(better_syn)
        answer = faq_answers[index]
        answer_dict["most_similar_question"] = faq_questions_raw[index]
        answer_dict["answer"] = answer
        answer_dict["score"] = max_cos	
        return answer_dict

def zero_test():
    synonims = []
    for quest in faq_questions_raw:
        answ = find_answer(quest)
        synonims.append(answ["most_similar_question"])
    print(np.mean(faq_questions_raw == synonims))
    return

def general_test():
    synonims = []
    stop_words = set(stopwords.words('english'))
    questions_copy = faq_questions_raw
    for quest in questions_copy:
        quest_token = word_tokenize(quest)
        filtered_sentence = [w for w in quest_token if not w in stop_words]
        text = ""
        for word in filtered_sentence:
            synonim = word
            k = 1
            while synonim == word:
                try:
                    syns = wn.synset(word + ".n.0" + str(k))
                    words = syns.name().split('.')
                    synonim = words[0]
                    k += 1
                except Exception:
                    break
            text += synonim + " "         
        answ = find_answer(text)
        synonims.append(answ["most_similar_question"])
    print(np.mean(faq_questions_raw == synonims))
    return

def get_question_groups(threshold):
    global model
    global faq_questions_raw
    global faq_questions
    threshold = float(threshold)
    dict_questions = {}   
    for quest in faq_questions:
        value = sent_feat(quest, model)
        questions = {}
        for quest2 in faq_questions:
            if quest != quest2:
                f_value = sent_feat(quest2, model)
                if cosine(value, f_value) > threshold:
                    questions[quest2] = cosine(value, f_value)
        if questions:
            sorted_questions = sorted(questions.items(), key=operator.itemgetter(1))
            dict_questions[quest] = [k[0] for k in sorted_questions]
    return dict_questions