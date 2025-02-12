# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:01:54 2025

@author: Awarri User
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB





X_train = np.array([
    
    [0, 1, 1],
    
    [0, 0, 1],
    
    [0, 0, 0],
    
    [1, 1, 0]]
    
    )

Y_train = ['Y', 'N', 'Y', 'Y']

X_test = np.array([[1, 1, 0]])


def get_label_indices(labels):
    
    """
... Group samples based on their labels and return indices
... @param labels: list of labels
... @return: dict, {class1: [indices], class2: [indices]}
... """
    
    from collections import defaultdict
    
    label_indices = defaultdict(list)
    
    for index, label in enumerate(labels):
        label_indices[label].append(index)
        
    return label_indices

label_indices = get_label_indices(Y_train)

print('label_indices:\n', label_indices)





def get_prior(label_indices):
    prior = {label: len(indices) for label, indices in label_indices.items()}
    
    total_count = sum(prior.values())
    
    for label in prior:
        prior[label] /= total_count
        
    return prior

prior = get_prior(label_indices)

print('Prior:', prior)




def get_likelihood(features, label_indices, smoothing=0):
    
    likelihood = {}
    for label, indices in label_indices.items():
        
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        
        total_count = len(indices)
        
        likelihood[label] = likelihood[label] /(total_count + 2 * smoothing)
        
    return likelihood


smoothing = 1
likelihood = get_likelihood(X_train, label_indices, smoothing)
print('Likelihood:\n', likelihood)



def get_posterior(X, prior, likelihood):
    
    
    """
... Compute posterior of testing samples, based on prior and
... likelihood
... @param X: testing samples
... @param prior: dictionary, with class label as key,
... corresponding prior as the value
... @param likelihood: dictionary, with class label as key,
... corresponding conditional probability
... vector as value
... @return: dictionary, with class label as key, correspondi
... posterior as value
... """

    posteriors = []
    
    for x in X:
       
       # Posterior is proportional to prior * likelihood
        posterior = prior.copy()

        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])

        sum_posterior = sum(posterior.values())  # Ensure this is outside the inner loop
        
        for label in posterior:
            if posterior[label] == float('inf'): posterior[label] = 1.0
            else:
                
                posterior[label] /= sum_posterior
            posteriors.append(posterior.copy())
                
    return posteriors

posterior = get_posterior(X_test, prior, likelihood)

print('Posterior:\n', posterior)








#### Scikit learn

clf = BernoulliNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)


pred_prob = clf.predict_proba(X_test)
print('[scikit-learn] Predicted probabilities:\n', pred_prob)


pred = clf.predict(X_test)
print('[scikit-learn] Prediction:', pred)






### real example

import numpy as np
import pandas as pd


data_path = r"ml-1m/ratings.dat"
df = pd.read_csv(data_path, header=None, sep='::', engine='python')
df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
print(df)

n_users = df['user_id'].nunique()
n_movies = df['movie_id'].nunique()
print(f"Number of users: {n_users}")
print(f"Number of movies: {n_movies}")




def load_user_rating_data(df, n_users, n_movies):
    data = np.zeros([n_users, n_movies], dtype=np.intc)
    movie_id_mapping = {}
    
    for user_id, movie_id, rating in zip(df['user_id'], df['movie_id'], df['rating']):
        user_id = int(user_id) - 1  # Adjust to zero-based index
        
        if movie_id not in movie_id_mapping:
            movie_id_mapping[movie_id] = len(movie_id_mapping)  # Assign new index
        
        data[user_id, movie_id_mapping[movie_id]] = int(rating)  # Store rating
    
    return data, movie_id_mapping

data, movie_id_mapping = load_user_rating_data(df, n_users, n_movies)

values, counts = np.unique(data, return_counts=True)

for value, count in zip(values, counts):
    print(f'Number of rating {value}: {count}')
    
print(df['movie_id'].value_counts())


target_movie_id = 2858
X_raw = np.delete(data, movie_id_mapping[target_movie_id], axis=1)
Y_raw = data[:, movie_id_mapping[target_movie_id]]
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)




recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples')



from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(len(Y_train), len(Y_test))

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])


prediction = clf.predict(X_test)

print(prediction[:10])

accuracy = clf.score(X_test, Y_test)

print(f'The accuracy is: {accuracy*100:.1f}%')

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, prediction, labels=[0, 1]))

#f1 score, recall, precision
from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(Y_test, prediction, pos_label=1)
recall_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=0)

#classification report
from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)

for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            # if truth and prediction are both 1
            if y == 1:
                true_pos[i] += 1
            # if truth is 0 while prediction is 1
            else:
                false_pos[i] += 1
        else:
            break




#graph

import matplotlib.pyplot as plt        
        
n_pos_test = (Y_test == 1).sum()        
n_neg_test = (Y_test == 0).sum()
true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]


plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#AUC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# K-fold settings
k = 5
k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Model hyperparameters
smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}

# K-fold cross-validation
for train_indices, test_indices in k_fold.split(X, Y):
    X_train_k, X_test_k = X[train_indices], X[test_indices]
    Y_train_k, Y_test_k = Y[train_indices], Y[test_indices]

    for alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record[alpha] = {}

        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train_k, Y_train_k)

            prediction_prob = clf.predict_proba(X_test_k)
            pos_prob = prediction_prob[:, 1]  # Probability for the positive class
            
            auc = roc_auc_score(Y_test_k, pos_prob)
            auc_record[alpha][fit_prior] = auc_record[alpha].get(fit_prior, 0.0) + auc

# Print the average AUC for each hyperparameter combination
for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        print(f'Alpha: {smoothing}, Fit Prior: {fit_prior}, AUC: {auc/k:.5f}')


clf = MultinomialNB(alpha=6.0, fit_prior=True)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('AUC with the best model:', roc_auc_score(Y_test, pos_prob))
