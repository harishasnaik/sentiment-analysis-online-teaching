# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import nltk
# nltk.download()
#call the nltk downloader
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
#%matplotlib inline

# %%
"""
## Reading Data

"""

# %%
df = pd.read_csv('main_data.csv')
df.head(10)

# %%
df = df.dropna()
df['Sentiment'] = ''
# df['Sentiment'] = np.where(df['Rating'] > 3, 'Pos')
# df['Sentiment'] = np.where(df['Rating'] < 3, 'Neg')
# df['Sentiment'] = np.where(df['Rating'] == 3, 'Neutral')
df.loc[df['Rating'] > 3, 'Sentiment'] = df.loc[df['Rating'] > 3, 'Sentiment'].replace('', 'Positive')
df.loc[df['Rating'] < 3, 'Sentiment'] = df.loc[df['Rating'] < 3, 'Sentiment'].replace('', 'Negative')
df.loc[df['Rating'] == 3, 'Sentiment'] = df.loc[df['Rating'] == 3, 'Sentiment'].replace('', 'Neutral')
df.head()

# %%
"""
## if you want to use only positive and negative class then use the code below and comment the one in above cell
"""

# %%
# df = df.dropna(inplace=True) #drop null 
# df = df[df['Rating'] != 3] #drop neutral rating
# #encode 4,5 as 1 for positive sentiment & 1,2 as 0 for negative sentiment
# df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)
# df.head()

# %%
seaborn.countplot(df['Sentiment'])

# %%
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
porter = PorterStemmer()
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        dede = porter.stem(no_non_ascii.strip())
        normalized_texts.append(dede)
    return normalized_texts
        
reviews = normalize_texts(df['Reviews'])

# %%
reviews = np.array(reviews[:500])
reviews.shape

# %%
len(reviews)

# %%
"""
## Making vectorizers
"""

# %%
corpus = {}
for doc in reviews:
    for w in doc.split():
        if w not in corpus:
            corpus[w]=len(corpus)

# %%
X = []
for doc in reviews:
    count = {}
    vector = np.zeros(len(corpus))
    for w in doc.split():
        if w not in count:
            count[w]=1
        else:
            count[w]+=1
    for k, v in count.items():
        index = corpus[k]
        vector[index]=v
    X.append(vector)

# %%
X3 = np.array(X)

# %%
"""
## Train Test Split
"""

# %%
X_train, X_test, y_train, y_test = train_test_split(X3, df['Sentiment'][:500], test_size=0.20,
                                                    random_state = 342, stratify = df['Sentiment'][:500])

# %%
seaborn.countplot(y_train)

# %%
"""
## Naive bayes
"""

# %%
parameters = {'alpha':[0.01,0.001,0.1,1, 0.05, 0.5] }
clf = MultinomialNB()
clf = GridSearchCV(clf, parameters, cv =5)
clf.fit(X_train, y_train)
accuracy12 = int((clf.best_score_) *100)
# %%
print("Best cross-validation accuracy: {:.2f}".format(clf.best_score_))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
print("Best parameters: {}".format(clf.best_params_))

# %%
y_pred_nv = clf.predict(X_test)
pos1 = 0
neg1 = 0
neu = 0
# %%
for i, j in zip(y_test, y_pred_nv):
    print("actual:", i, " pred:", j)
    if i == 'Positive':
        pos1 = pos1+1
    elif i == 'Negative':
        neg1=neg1+1
    elif i == 'Neutral':
        neu = neu+1

print('Positive Reviews',pos1)
print('Negative Reviews',neg1)
print('Neutral Reviews',neu)

# %%
conf_mat = confusion_matrix(y_test, y_pred_nv)
df_cm = pd.DataFrame(conf_mat, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,10))
seaborn.heatmap(df_cm,cmap= "YlGnBu", annot=True,annot_kws={"size": 16})
# # plt.savefig("bayes confusion")

# %%
#print(classification_report(y_test, y_pred_nv))

# %%
"""
## Random Forest
"""

# %%






parameters = {'n_estimators':[10,100,200],  'max_depth':[2,3,4]}

clf = RandomForestClassifier()

clf = GridSearchCV(clf, parameters, cv =3)

clf.fit(X_train, y_train)
RF_accuracy = int((clf.best_score_)*100)

#RF_accuracy = 89

# %%
print("Best cross-validation accuracy: {:.2f}".format(clf.best_score_))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
print("Best parameters: {}".format(clf.best_params_))

# %%
y_pred_rf = clf.predict(X_test)
for i, j in zip(y_test, y_pred_rf):
    print("actual:", i, " pred:", j)

# %%
conf_mat = confusion_matrix(y_test, y_pred_rf)
df_cm = pd.DataFrame(conf_mat, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,10))
seaborn.heatmap(df_cm,cmap= "YlGnBu", annot=True,annot_kws={"size": 16})

# %%
#print(classification_report(y_test, y_pred_rf))

import matplotlib.pyplot as plt
x = ['Positive', 'Negative', 'Neutral']
energy = [pos1, neg1, neu]
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, energy, color='green')
plt.xlabel("Algorithms")
plt.ylabel("Accuracy(%")
plt.title("Accuracy of Algorithms IDE")
plt.xticks(x_pos, x)

plt.show()



'''


import matplotlib.pyplot as plt
x = ['Naive bayes', 'RF']
energy = [accuracy12, RF_accuracy]
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, energy, color='green')
plt.xlabel("Algorithms")
plt.ylabel("Accuracy(%")
plt.title("Accuracy of Algorithms IDE")
plt.xticks(x_pos, x)
# y = [svmaccuracy, 0, 0]
# plt.title('Accuracy')
# plt.bar(x, y)
plt.show()

'''

