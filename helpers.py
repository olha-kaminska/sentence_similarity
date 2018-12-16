from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import numpy as np
import string


''' function delete_stopwords(sentence) 
input: one sentence as string, output: this sentence withot stopwords as string
functional: delete standart inglish stopword from one sentence
using libraries: stopwords, word_tokenize
example:
    input: 'This is a sample sentence, showing off the stop words filtration.'
    output: 'This sample sentence , showing stop words filtration .'
    '''    
def delete_stopwords(sentence):
    sentence_token = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_token = [w for w in sentence_token if not w in stop_words]
    filtered_sentence = ' '.join(filtered_token)
    return filtered_sentence

''' function get_synonims(sentence_list) 
input: list of the sentences, output: list of the similar sentence
functional: for each sentence from input list delete stopwords, 
            for other words - looking for the closest by meaning synonim, if exist - replace original word by this synonim
using libraries: wordnet, word_tokenize
example:
    input: ['This is an example of sentence.', 'One more sentence in the list.']
    output: ['This model conviction.', 'One conviction tilt.']
    ''' 
def get_synonims(sentence_list):
    synonims_list = []    
    for quest in sentence_list:        
        filtered_sentence = delete_stopwords(quest)
        filtered_token = word_tokenize(filtered_sentence)
        text = ""
        for word in filtered_token:
            synonim = word
            counter = 1
            while synonim == word:
                try:
                    closest_synonim = wn.synset(word + ".n.0" + str(counter))
                    words = closest_synonim.name().split('.')
                    synonim = words[0].replace('_', ' ')
                    counter += 1
                except Exception:
                    break
            text += synonim + " "
        synonims_list.append(text)
    return synonims_list

''' function sentence_to_vec(sent, model)
input: sentence (as string) and model, output: vector
functional: transform sentence to the 300-dimension vector, based on input model
using libraries: numpy
recommended model: GoogleNews-vectors-negative300
example:
    input: 'This is an example of sentence.', model_word2vec
    output: [0.44, 0.33, ..., 0.12] (300 values)
    ''' 
def sentence_to_vec(sent, model):
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

''' function normalization(raw_string)
input: sentence as string, output: sentence without punctuation with words in lowercase
functional: delete punctuation in sentence, write all words in lowercase
using libraries: string
example:
    input: 'This is an example, an Example of the sentence.'
    output: 'this is an example an example of the sentence'
    ''' 
def normalization(raw_string):
    exclude = string.punctuation
    new_string = ''.join(ch for ch in raw_string if ch not in exclude)
    new_string = new_string.lower()
    return new_string