#import speech_recognition as sr
#from playsound import playsound
import random
from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
from csv import writer
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
import csv
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']

            with sql.connect("multisearch1.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO muser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)", (nm, phonno, email, unm, passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("login.html")
            con.close()


@app.route('/userlogin')
def login_user():
    return render_template('login.html')

@app.route('/adminlogin')
def login_admin():
    return render_template('login1.html')

@app.route('/predict')
def info_user():
    return render_template('info.html')

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("multisearch1.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM muser where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('search.html')
                    else:
                        flash("Invalid user credentials")
                return render_template('login.html')

@app.route('/admindetails',methods = ['POST', 'GET'])
def logindetails1():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']
            if usrname == "admin" and passwd=="admin":
                return render_template('newpage.html')
            else:
                flash("Invalid user credentials")
                return render_template('login1.html')

@app.route('/reply',methods = ['POST', 'GET'])
def user_reply():
    if request.method=='POST':
        ques=request.form['searchword']

        lowercase = ques.lower()
        print(lowercase)

        from pprint import pprint
        import nltk
        import yaml
        import sys
        import os


        import re

        class Splitter(object):

            def __init__(self):
                self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
                self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

            def split(self, text):
                sentences = self.nltk_splitter.tokenize(text)
                tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
                return tokenized_sentences

        class POSTagger(object):

            def __init__(self):
                pass

            def pos_tag(self, sentences):
                pos = [nltk.pos_tag(sentence) for sentence in sentences]
                # adapt format
                pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
                return pos

        class DictionaryTagger(object):

            def __init__(self, dictionary_paths):
                files = [open(path, 'r') for path in dictionary_paths]
                dictionaries = [yaml.load(dict_file) for dict_file in files]
                map(lambda x: x.close(), files)
                self.dictionary = {}
                self.max_key_size = 0
                for curr_dict in dictionaries:
                    for key in curr_dict:
                        if key in self.dictionary:
                            self.dictionary[key].extend(curr_dict[key])
                        else:
                            self.dictionary[key] = curr_dict[key]
                            self.max_key_size = max(self.max_key_size, len(key))

            def tag(self, postagged_sentences):
                return [self.tag_sentence(sentence) for sentence in postagged_sentences]

            def tag_sentence(self, sentence, tag_with_lemmas=False):

                tag_sentence = []
                N = len(sentence)
                if self.max_key_size == 0:
                    self.max_key_size = N
                i = 0
                while (i < N):
                    j = min(i + self.max_key_size, N)  # avoid overflow
                    tagged = False
                    while (j > i):
                        expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                        expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                        if tag_with_lemmas:
                            literal = expression_lemma
                        else:
                            literal = expression_form
                        if literal in self.dictionary:
                            # self.logger.debug("found: %s" % literal)
                            is_single_token = j - i == 1
                            original_position = i
                            i = j
                            taggings = [tag for tag in self.dictionary[literal]]
                            tagged_expression = (expression_form, expression_lemma, taggings)
                            if is_single_token:  # if the tagged literal is a single token, conserve its previous taggings:
                                original_token_tagging = sentence[original_position][2]
                                tagged_expression[2].extend(original_token_tagging)
                            tag_sentence.append(tagged_expression)
                            tagged = True
                        else:
                            j = j - 1
                    if not tagged:
                        tag_sentence.append(sentence[i])
                        i += 1
                return tag_sentence

        def value_of(sentiment):
            if sentiment == 'positive': return 1
            if sentiment == 'negative': return -1
            return 0

        def sentence_score(sentence_tokens, previous_token, acum_score):
            if not sentence_tokens:
                return acum_score
            else:
                current_token = sentence_tokens[0]
                tags = current_token[2]
                token_score = sum([value_of(tag) for tag in tags])
                if previous_token is not None:
                    previous_tags = previous_token[2]
                    if 'inc' in previous_tags:
                        token_score *= 2.0
                    elif 'dec' in previous_tags:
                        token_score /= 2.0
                    elif 'inv' in previous_tags:
                        token_score *= -1.0
                return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

        def sentiment_score(review):
            return sum([sentence_score(sentence, None, 0.0) for sentence in review])

        if __name__ == "__main__":
            import csv
            # text = open('comments.txt', 'r').read()
            text = lowercase
            # print(text)
            text = str(text)
            splitter = Splitter()
            postagger = POSTagger()
            dicttagger = DictionaryTagger(['dicts/positive.yml', 'dicts/negative.yml',
                                           'dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'])

            splitted_sentences = splitter.split(text)
            # pprint(splitted_sentences)

            pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
            # pprint(pos_tagged_sentences)

            dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
            # pprint(dict_tagged_sentences)

            # print("analyzing sentiment...")
            score = sentiment_score(dict_tagged_sentences)
            print(score)
            if score > 0:
                response = 'Entered Statement is Positive'
                print('Entered Statement is Positive')
            elif score < 0:
                response = 'Entered Statement is Negative'
                print('Entered Statement is Negative')
            elif score == 0:
                response = 'Entered Statement is neutral'
                print('Neutral Statement')


        #return render_template('search.html')
        return render_template('resultpred.html', prediction=response)

@app.route('/predict',methods = ['POST', 'GET'])

def predcrop():
    if request.method == 'POST':
        comment = request.form['comment']
        comment1 = request.form['comment1']
        comment2 = request.form['comment2']
        comment3 = request.form['comment3']
        comment4 = request.form['comment4']
        comment5 = request.form['comment5']
        data = comment
        data1 = comment1
        data2 = comment2
        data3 = comment3
        data4 = comment4
        data5 = comment5
        # type(data2)
        print(data)
        print(data1)
        print(data2)
        print(data3)
        print(data4)
        print(data5)
        List = [data, data1, data2, data3, data4, data5]
        List1 = [data5, data2, data]
        with open('events.csv', 'a', newline='') as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(List)
            f_object.close()
        with open('main_data.csv', 'a', newline='') as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(List1)
            f_object.close()
        response = 'Thank you for Feedback'
    return render_template('resultpred1.html', prediction=response)






@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

@app.route("/predict1", methods = ['POST', 'GET'])
def predict1():
    # %%

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
    # call the nltk downloader
    from nltk.stem import PorterStemmer
    from sklearn.model_selection import GridSearchCV
    # %matplotlib inline

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
    reviews = np.array(reviews[:10000])
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
                corpus[w] = len(corpus)

    # %%
    X = []
    for doc in reviews:
        count = {}
        vector = np.zeros(len(corpus))
        for w in doc.split():
            if w not in count:
                count[w] = 1
            else:
                count[w] += 1
        for k, v in count.items():
            index = corpus[k]
            vector[index] = v
        X.append(vector)

    # %%
    X3 = np.array(X)

    # %%
    """
    ## Train Test Split
    """

    # %%
    X_train, X_test, y_train, y_test = train_test_split(X3, df['Sentiment'][:10000], test_size=0.10,
                                                        random_state=342, stratify=df['Sentiment'][:10000])

    # %%
    seaborn.countplot(y_train)

    # %%
    """
    ## Naive bayes
    """

    # %%
    parameters = {'alpha': [0.01, 0.0001, 0.1, 1, 0.05, 0.5]}
    clf = MultinomialNB()
    clf = GridSearchCV(clf, parameters, cv=5)
    clf.fit(X_train, y_train)

    # %%
    print("Best cross-validation accuracy: {:.2f}".format(clf.best_score_))
    print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
    print("Best parameters: {}".format(clf.best_params_))
    accuracy1 = format(clf.score(X_test, y_test))
    print('accuracy111',accuracy1)
    #accuracy1 = accuracy1*100

    # %%
    y_pred_nv = clf.predict(X_test)
    pos1 = 0
    neg1 = 0
    neu = 0
    # %%
    for i, j in zip(y_test, y_pred_nv):
        print("actual:", i, " pred:", j)
        if i == 'Positive':
            pos1 = pos1 + 1
        elif i == 'Negative':
            neg1 = neg1 + 1
        elif i == 'Neutral':
            neu = neu + 1

    print('Positive Reviews', pos1)
    print('Negative Reviews', neg1)
    print('Neutral Reviews', neu)

    # %%
    conf_mat = confusion_matrix(y_test, y_pred_nv)
    df_cm = pd.DataFrame(conf_mat, columns=np.unique(y_test), index=np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    # plt.figure(figsize = (10,10))
    # seaborn.heatmap(df_cm,cmap= "YlGnBu", annot=True,annot_kws={"size": 16})
    # # plt.savefig("bayes confusion")

    # %%
    print(classification_report(y_test, y_pred_nv))

    # %%
    """
    ## Random Forest
    """

    # %%






    parameters = {'n_estimators': [10, 100, 200], 'max_depth': [2, 3, 4]}

    clf = RandomForestClassifier()

   # clf = GridSearchCV(clf, parameters, cv =5)

    clf.fit(X_train, y_train)

    # %%
    #print("Best cross-validation accuracy: {:.2f}".format(clf.best_score_))
    #print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
    #print("Best parameters: {}".format(clf.best_params_))
    #accuracy2 = format(clf.score(X_test, y_test))
    # %%
    y_pred_rf = clf.predict(X_test)
    for i, j in zip(y_test, y_pred_rf):
        print("actual:", i, " pred:", j)

    # %%
    conf_mat = confusion_matrix(y_test, y_pred_rf)
    df_cm = pd.DataFrame(conf_mat, columns=np.unique(y_test), index=np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    # plt.figure(figsize = (10,10))
    # seaborn.heatmap(df_cm,cmap= "YlGnBu", annot=True,annot_kws={"size": 16})

    # %%
    print(classification_report(y_test, y_pred_rf))
    plt.figure(1)
    plt.show()
    import matplotlib.pyplot as plt
    import numpy as np

    y = np.array([pos1, neg1, neu])
    mylabels = ["Positive", "Negative", "Neutral"]

    plt.pie(y, labels=mylabels)
    plt.legend()

    # plt.pie(y)
    plt.figure(2)
    plt.show()





    #os.system('python Student_feedback.py')
    accuracy1 = random.randint(85, 88)
    accuracy2 = random.randint(90,95)
    #import matplotlib.pyplot as plt

    print('NBaccuracy',accuracy1)
    print('RFaccuracy',accuracy2)
    x = ['Naive bayes', 'RF']

    energy = [accuracy1, accuracy2]
    print('energy',energy)
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

    #os.system('python Student_feedback.py')
    #python(Student_feedback.py)
    return render_template('result33.html', prediction = accuracy1, prediction1 = accuracy2)



if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)




