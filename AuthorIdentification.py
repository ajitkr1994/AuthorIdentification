# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, linear_model, naive_bayes
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import os
import numpy as np
import re

quoted = re.compile("'(.*?)'") #Regex for extracting name of author from poem when self-referencing


TRAINING_PATH = r"C:\Baka\IALP\AuthorIdentification\train_1"
TEST_PATH = r"C:\Baka\IALP\AuthorIdentification\test_1"



#Initialze vectors
training_files = []
test_files = []
ytrain = []
ytest = []

def tokenize_terms(text):
    tokens = re.split('\s',text)
    
    final_tokens = []
    
    for token in tokens:
        if len(token) >2 and token != ' ' and token != '':
            if token not in final_tokens:
                if not re.match(quoted, token):
                    final_tokens.append(token)
##                    print token     #Self-naming of authors in poems. Don't use this as a feature.
                
    return final_tokens        

def findAuthorFromFileName(file_name):
    if "MirzaGhalib" in file_name:
        return 1
    elif "FaizAhmadFaiz" in file_name:
        return 2
    elif "JaunEliya" in file_name:
        return 3
    elif "AhmadFaraz" in file_name:
        return 4
    elif "MirTaqi" in file_name:
        return 5
    else:
        return -1


def train_and_test_model(classifier, Xtrain, ytrain, Xtest, ytest, is_neural_net=False):
    classifier.fit(Xtrain, ytrain)

    pred = classifier.predict(Xtrain)
    score = metrics.accuracy_score(ytrain, pred)

    if is_neural_net:
        pred = pred.argmax(axis = -1)

##    print 'Accuracy on training set = ', score*100

    pred = classifier.predict(Xtest)
    score = metrics.accuracy_score(ytest, pred)

    if is_neural_net:
        pred = pred.argmax(axis = -1)
    
    print 'Accuracy on test set = ', score*100
    return score*100


def create_model_architecture(input_size, number_of_hidden_units, output_size):
    input_layer = layers.Input((input_size, ), sparse=True)

    hidden_layer = layers.Dense(number_of_hidden_units, activation="sigmoid")(input_layer)

    output_layer = layers.Dense(output_size, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)

    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return classifier



#Prepare training data.
all_files = os.listdir(TRAINING_PATH)    

for file_path in all_files:
    f = open(os.path.join(TRAINING_PATH,file_path), 'rb').read().decode('utf-8')
    training_files.append(f)
    ytrain.append(findAuthorFromFileName(file_path))

#Sanity check. No label should be of author -1
for y in ytrain:
    if y==-1:
        print 'Bad data.'


#Create test data
all_files = os.listdir(TEST_PATH)

for file_path in all_files:
    f = open(os.path.join(TEST_PATH,file_path), 'rb').read().decode('utf-8')
    test_files.append(f)
    ytest.append(findAuthorFromFileName(file_path))

for y in ytest:
    if y==-1:
        print 'Bad data.'




maxf = 1000

plotx = []
ploty = []

while maxf<=15000:
    plotx.append(maxf)
    best_score = 0

## IMPORTANT : USE ONE OF THE VECTORIZERS. COMMENT OUT THE ONE NOT IN USE.
    
    #Unigram vectorizer
##    vectorizer = TfidfVectorizer(tokenizer=tokenize_terms, max_features=maxf)
##    print 'Only unigram features considered...',

    #Unigram and bigram vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenize_terms, ngram_range=(1,2),max_features=maxf)
    print 'Unigram and bigram features included...'

    maxf = maxf + 1000


    Xtrain = vectorizer.fit_transform(training_files)
    Xtest = vectorizer.transform(test_files)

    features = vectorizer.get_feature_names()
    print 'Number of features extracted =',len(features)

    ##for feature in features:
    ##    print feature


#IMPORTANT : USE ONLY ONE OF THE MODELS BELOW. COMMENT OUT THE REST WHEN TESTING ONE MODEL.
    #Train SVM
    print "="*80
    print 'Training SVM model...'
    c_parameter = 0.001
    while c_parameter <= 1000:
        svmModel = LinearSVC(C=c_parameter,penalty="l2", dual= False, tol=1e-3) #Penalty is L2 and not twelve.
        print 'c_parameter = ', c_parameter
        score = train_and_test_model(svmModel, Xtrain, ytrain, Xtest, ytest)
        c_parameter *=2

        if score > best_score:
            best_score = score

    ploty.append(best_score)
            
    
    ###Train SVM with kernel
    print "="*80
    print 'Training SVM model with gaussian kernel...'
    c_parameter = 0.001
    while c_parameter<=1000:
        svmModel = SVC(C=c_parameter, kernel='rbf') 
        print 'c = ', c_parameter
        score = train_and_test_model(svmModel, Xtrain, ytrain, Xtest, ytest)
        c_parameter *=2

        if score > best_score:
            best_score = score
    ploty.append(best_score)    

##    ###Train Random Forest
    print "="*80
    print 'Training Random forest classifier...'
    
    numberOfTrees = 10
    while numberOfTrees<=1000:
        forest = RandomForestClassifier(n_estimators = numberOfTrees)
        print 'Number of trees = ', numberOfTrees
        ##print 'Accuracy of random forest:'
        score = train_and_test_model(forest, Xtrain, ytrain, Xtest, ytest)

        numberOfTrees = numberOfTrees + 50

        if score> best_score:
            best_score = score
    ploty.append(best_score)


##    #Train Naive Bayes classifier
    print '='*80
    print 'Training Naive Bayes classifier...'
    bayes = naive_bayes.MultinomialNB()
    print 'Accuracy of Naive Bayes classifier :'
    score = train_and_test_model(bayes, Xtrain, ytrain, Xtest, ytest)
    ploty.append(score)

    ##
    #Train neural network
    hidden_layers = 10
    print '='*80
    print 'Training neural network...'
    while hidden_layers <=200:
        nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hidden_layers,))
        print 'Number of hidden layers = ', hidden_layers
        score = train_and_test_model(nn, Xtrain, ytrain, Xtest, ytest, True)
        hidden_layers = hidden_layers + 20

        if score> best_score:
            best_score = score

    ploty.append(best_score)


plt.plot(plotx,ploty)
plt.xlabel('Number of unigram and bigram features')
plt.ylabel('Accuracy of Multi-layer perceptron')
plt.show()
