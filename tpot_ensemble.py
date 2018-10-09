# tpot ensemble test!

# user input
num_pipes = 20

# regular stuff
import numpy as np 
import pandas as pd 
import os

# import TPOT and sklearn stuff
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import VotingClassifier

# create train and test sets
digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75, test_size=0.25)
tpot = TPOTClassifier(verbosity=2, 
                      scoring="accuracy",  
                      n_jobs=-1, 
                      generations=20, 
                      population_size=30)

scores = []
winning_pipes = []
tpots = []

# run three iterations
for i in range(num_pipes):

	model_tup = ('{}'.format(i + 1), tpot)
	tpots.append(model_tup)


# let's see if we can just do VotingClassifier
ens = VotingClassifier(tpots, 'hard')
ens.fit(X_train, y_train)
s = ens.score(X_test, y_test)

print('')
print('ensemble score = {}'.format(s))



