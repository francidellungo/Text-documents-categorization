from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
from curve_plot import plot_curve
from sklearn.linear_model import Perceptron



def Reuters_classification(classif = ''):
    documents = reuters.fileids()
    # List of document ids
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    cats = reuters.categories()

    # select 10 most common categories
    top10_categories = get_10most_common_categories(train_docs_id, cats)

    # create new dataset
    print 'create new dataset:'
    new_train_docs_id, y__train, new_test_docs_id, y__test = create_new_dataset(train_docs_id, test_docs_id,
                                                                                top10_categories)

    data_train = [reuters.raw(doc_id) for doc_id in new_train_docs_id]
    data_test = [reuters.raw(doc_id) for doc_id in new_test_docs_id]

    vectorizer = TfidfVectorizer(stop_words='english')

    x_train = vectorizer.fit_transform(
        data_train)  # Learn vocabulary and idf, return term-document matrix, is equivalent to fit followed by transform
    x_test = vectorizer.transform(data_test)

    print 'x_train dimension', x_train.shape
    print 'x_test dimension ', x_test.shape

    if classif == 'bayes':
        #clf = MultinomialNB(alpha=1, fit_prior=False)
        clf = BernoulliNB(fit_prior=False, binarize=.05, alpha=.15)
    elif classif == 'perceptron':
        clf = Perceptron(n_iter=50, n_jobs=-1, alpha=1)
    else:
        print 'errore'

    clf.fit(x_train, y__train)
    pred = clf.predict(x_test)
    score = metrics.accuracy_score(y__test, pred)
    print('clf: ', classif, " accuracy:   %0.3f" % score)

    print('calcolo learning curve ')
    x = x_train
    y = y__train
    cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=40)
    t_sizes = np.logspace(np.log10(.05), np.log10(1.0), 10)

    train_sizes, train_scores, test_scores = learning_curve(
        clf, x, y, cv=cv, n_jobs=-1, train_sizes=t_sizes)

    if classif == 'bayes':
        plot_curve(train_sizes, train_scores, test_scores,
                   title="Naive Bayes Reuters-21578")
    elif classif == 'perceptron':
        plot_curve(train_sizes, train_scores, test_scores,
                   title="Perceptron Reuters-21578")



def get_10most_common_categories(train_docs_id, cats):
    y_train = []
    for t in train_docs_id:
        y_train.append(reuters.categories(t))
    print 'y train: ', y_train
    print y_train

    y_train_split = []
    for train in range(0, len(y_train)):
        for i in range(0, len(y_train[train])):
            y_train_split.append(y_train[train][i])

    count_list = []
    for category in cats:
        instances = y_train_split.count(category)
        count_list.append((instances, category))

    count_list.sort(reverse=True)
    top10_categories = []
    for i in range(0, 10):
        top10_categories.append(count_list[i])
    return top10_categories

def create_new_dataset(train_docs_id, test_docs_id, top10_categories):
    #train data
    new_train_docs_id = []
    y__train = []
    instaces_vector = [0, 0] #vettore per contare istanze di earn e di acq
    for train_doc in train_docs_id:
        max_instances = float('inf')
        new_category = ''
        for doc_category in reuters.categories(train_doc):
            for i in range(0, len(top10_categories)):
                if doc_category == top10_categories[i][1] and max_instances > top10_categories[i][0]:
                    new_category = doc_category
                    max_instances = top10_categories[i][0]
        if new_category != '': #se il documento ha almeno una classe appartenente alla top10
            if new_category == u'earn':
                if instaces_vector[0] < 500:
                    instaces_vector[0] += 1
                    new_train_docs_id.append(train_doc)
                    y__train.append(new_category)
            elif new_category == u'acq':
                if instaces_vector[1] < 500:
                    instaces_vector[1] += 1
                    new_train_docs_id.append(train_doc)
                    y__train.append(new_category)
            else:
                new_train_docs_id.append(train_doc)
                y__train.append(new_category)

    # test data
    new_test_docs_id = []
    y__test = []
    instaces_vector = [0, 0]
    for test_doc in test_docs_id:
        max_instances = float('inf')
        new_category = ''
        for doc_category in reuters.categories(test_doc): #ciclo sui documenti

            for i in range(0, len(top10_categories)):
                if doc_category in top10_categories[i][1] and max_instances > top10_categories[i][0]:
                    new_category = doc_category
                    max_instances = top10_categories[i][0]
            if new_category != '':  # se il documento ha almeno una classe appartenente alla top10
                new_test_docs_id.append(test_doc)
                y__test.append(new_category)


    return new_train_docs_id, y__train, new_test_docs_id, y__test

