import pandas as pd
import fasttext
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

data = pd.read_csv("data.csv")

#preprocessing

#print(data.head())
#print(data.dtypes)
#print(data.isnull().sum())
data['Intent'] = data['Intent'].str.replace('[', '')
data['Intent'] = data['Intent'].str.replace(']', '')
data['Intent'] = pd.to_numeric(data['Intent'], errors='coerce')
data['Data'] = data['Data'].astype('str')

# save data in suitable format

indent = []
text = []
with open('train_data.txt', 'w') as f:
    for i in range(len(data.index)):
        indent.append('__label__' + str(data.loc[data.index[i], 'Intent']))
        text.append(data.loc[data.index[i], 'Data'].replace('\r', ''))
        f.write('__label__' + str(data.loc[data.index[i], 'Intent']) + ' ' + data.loc[data.index[i], 'Data'] + '\n')

# make test set
testset = {}
for i in range(len(data.index)):
    quest = data.loc[data.index[i], 'Data'].replace('\r', '')
    quest_token = word_tokenize(quest)
    text = ""
    for word in quest_token:
        synonim = word
        k = 1
        while synonim == word:
            try:
                syns = wn.synset(word + ".n.0" + str(k))
                words = syns.name().split('.')
                synonim = words[0].replace('_', '')
                k += 1
            except Exception:
                break
        text += synonim + " "
    testset[text] = data.loc[data.index[i], 'Intent']

indent_test = []
text_test = []
with open('test_data.txt', 'w') as f:
    for i in testset.keys():
        indent_test.append('__label__' + str(testset[i]))
        text_test.append(i.replace('\r', ''))
        f.write('__label__' + str(testset[i]) + ' ' + i + '\n')

# model
             
model = fasttext.FastText.train_supervised('Trainset.txt')
print(model.test('test_data.txt')) # output: size, precision, recall

# to calculate accuracy

c = 0
for i in range(len(text_test)):
    c += int(model.predict(text_test[i])[0][0] == indent_test[i])
print(float(c) / len(text_test))   