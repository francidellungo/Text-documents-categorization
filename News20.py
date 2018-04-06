
import numpy as np
from curve_plot import plot_curve

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit



def News20_classification(classif=''):
    print('Loading 20 newsgroups dataset for categories')
    remove = ('headers', 'footers', 'quotes')

    data_train = fetch_20newsgroups(subset='train', categories=None,
                                    # Load the filenames and data of the train from the 20 newsgroups dataset
                                    shuffle=True, random_state=42,
                                    remove=remove)

    data_test = fetch_20newsgroups(subset='test', categories=None,
                                   # Load the filenames and data of the test from the 20 newsgroups dataset
                                   shuffle=True, random_state=42,
                                   remove=remove)

    # split target of training set and test set
    y_train, y_test = data_train.target, data_test.target

    vectorizer = TfidfVectorizer(stop_words='english')
    x_train = vectorizer.fit_transform(data_train.data)  # Learn vocabulary and idf,
                                          # return term-document matrix, is equivalent to fit followed by transform
    x_test = vectorizer.transform(data_test.data)
    print('x_train dimension', x_train.shape)
    print('x_test dimension ', x_test.shape)

    if classif == 'bayes':
        #clf = MultinomialNB(alpha=1, fit_prior=False)
         clf = BernoulliNB(fit_prior=False, binarize=0.05, alpha=.10)
    elif classif == 'perceptron':
        clf = Perceptron(n_iter=40, n_jobs=-1)
    else:
        print 'errore'

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    score = metrics.accuracy_score(y_test, pred)
    print('clf: ', classif, " accuracy:   %0.3f" % score)

    print('calcolo learning curve')
    x = x_train
    y = y_train
    cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=42)
    t_sizes = np.logspace(np.log10(.05), np.log10(1.0), 8)

    train_sizes, train_scores, test_scores = learning_curve(
        clf, x, y, cv=cv, n_jobs=-1, train_sizes=t_sizes)
    print('plot learning curve')
    if classif == 'bayes':
        plot_curve(train_sizes, train_scores, test_scores,
               title="Naive Bayes NewsGroups20")
    elif classif == 'perceptron':
        plot_curve(train_sizes, train_scores, test_scores,
                   title="Perceptron NewsGroups20")
