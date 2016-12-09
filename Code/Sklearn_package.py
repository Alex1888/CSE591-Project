import numpy as np
import sys
import scipy.stats
import csv
import math
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB



def Gaussian_Text(X,Y,testcase):
  clf = GaussianNB()
  clf.fit(X, Y)
  GaussianNB()
  result = clf.predict(testcase)
  return result

def MultinomialNB_Text(X,Y,testcase):
  clf = MultinomialNB()
  clf.fit(X, Y)
  MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
  result = clf.predict(testcase)
  return result

def BernoulliNB_Text(X,Y,testcase):
  clf = BernoulliNB()
  clf.fit(X, Y)
  BernoulliNB(alpha=1.0, class_prior=None, fit_prior=True)
  result = clf.predict(testcase)
  return result




def paired_t_test(X, Y, K, alpha, isoutputresult = True):
    """
    return (bound, z_score, accept/reject)
    """
    n = len(X)
    diff = np.zeros(K)
    for i in range(K):
        # 1 compute training set X_i using bootstrap resampling
        sample_index = np.random.randint(n, size=n)
        X_i = X[sample_index]
        Y_i = Y[sample_index]

        # 2 train both full and naive Bayes on sample X_i
        classes = classes = np.unique(Y_i)

        # 3 compute testing set X - X_i
        X_test = []
        Y_test = []
        for j in range(X.shape[0]):
          if j not in sample_index:
            X_test.append(X[j])
            Y_test.append(Y[j])
        X_test = np.array(X_test)  # correct data is in Y_test
        Y_test = np.array(Y_test)
        # 4 assess both on X - X_i
        Y_result = []; Y_result_naive = [];
        Y_result = BernoulliNB_Text(X_i, Y_i, X_test)

        num_err_full = 0
        num_err_naive = 0

        for y in range(len(Y_result)):
          index = Y_result[y]
          if classes[index] != Y_test[y]:
            num_err_full += 1
        if isoutputresult:
          print('sample, error_rate, samplenumber:', i, num_err_full, len(X_test))
        err_rate_full = num_err_full
        err_rate_naive = num_err_naive

        # 5 compute difference in error rates
        diff[i] = err_rate_full - err_rate_naive
    if isoutputresult:   
      print('all differences:'); print(diff)
    # compute mean, variance, and z-score
    mean_u = np.mean(diff)
    square_u = np.var(diff) 
    z_score = math.sqrt(K) * mean_u / square_u

    if isoutputresult:
      print('z-score:', z_score)

    # compute interval bound using inverse survival function of t distribution
    bound = scipy.stats.t.isf((1-alpha)/2.0, K-1)
    if isoutputresult:
      print('bound:', bound)

    # output conclusion based on tests
    if -bound < z_score < bound:
      if isoutputresult:
        print('accept: classifiers have similar performance')
      return (bound, z_score, 'accept')
    else:
      if isoutputresult:
        print('reject: classifiers have significantly different performance')
      return (bound, z_score, 'reject')

def load_data(filename):
    """
    Load data from files and return them as numpy arrays
    """

    fileptr = open(filename, "rU")
    lines = [line.rstrip() for line in fileptr]
    mlist = []
    for line in lines:
      mlist.append(line.split(","))
    mlist = np.array(mlist)
    return mlist

def prepare_data(dataset):
  "insert the front 4 column to X, and the last column to Y"
  X = []
  Y = []
  colomn_num = dataset.shape[1]
  for d in dataset:
     X.append(d[0:colomn_num-1].astype(np.float))
     Y.append(d[colomn_num-1])
  X = np.array(X)
  Y = np.array(Y)
  return (X, Y)


dataset = load_data('D:\\Courses\\591-Programming Complex Algorithms\\project\\dressing2.csv')
(X, Y) = prepare_data(dataset)
K = 30
alpha = 0.95
acceptnum = 0
times = 40
start_time = time.time()
for i in range(times):
  (bound, z_score, result)= paired_t_test(X, Y, K, alpha, False)
  print("[%d]: bound:%f, z_score:%f, %s" %(i, bound ,z_score ,result ))
  if result == 'accept':
    acceptnum += 1

print('Total:')
print("accept: %d times"  %acceptnum )
print("reject: %d times"   %(times- acceptnum))
print("---running time: %s seconds ---" % (time.time() - start_time))