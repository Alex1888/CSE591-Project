import sys
import numpy as np
import scipy.stats
import csv
import math

def bayes_params(X, Y):
    # 1 classess
    classes = np.unique(Y)

    # 2 Pc
    Pc = []
    for cla in classes:
      count = len(Y[Y==cla])
      Pc.append(count / Y.size)
    Pc = np.array(Pc)

    # 3 mean and covariance
    mean = []
    cov = []; cov_naive = []
    Di = create_D_matrix(classes, X, Y)
    for i in range(Di.shape[0]):
      di = Di[i]
      ni = di.shape[0]
      # ui is 1*4 matrix?di is 50*4 matrix
      ui = np.mean(di, axis = 0)[np.newaxis]
      mean.append(ui)
      n1 = np.ones(shape=(ni, 1))  #n1 is 50*1 matrix
      zi = di - np.dot(n1, ui)
      ziT = zi.T
      #ziT = np.transpose(zi)
      covar = np.dot(ziT, zi)   # di.shape[0] is ni len
      covar /= di.shape[0]
      #d = di.shape[1] #d is 4
      cov.append(covar)
      ##covar_naive = np.cov(ziT)
      cov_naive.append(np.diag(np.diag(covar)))
           
    mean = np.array(mean)
    cov = np.array(cov)
    cov_naive = np.array(cov_naive)

    return (classes, Pc, mean, cov, cov_naive)

def multi_var_normal_pdf(x, mean, cov, isNaive = False):
   """
   calculate the pdf valuea for the input case x
   """
   pbs = []
   for i in range(mean.shape[0]):
     pb = calculateProbability(x, mean[i], cov[i],isNaive)
     pbs.append(pb)
   return pbs

def bayes_test(x, classes, Pc, mean, cov, isNaive = False):
    """
    Do the Bayes test
    reutrn: the bayes values for each type
    """
    pdf = multi_var_normal_pdf(x, mean, cov, isNaive)
    y = []
    for i in range(Pc.shape[0]):
      y.append(pdf[i] * Pc[i])
    Y = np.argmax(y, axis = 0)
    return Y
      
def paired_t_test(X, Y, K, alpha):
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
        classes, Pc, mean, cov, cov_naive = bayes_params(X_i, Y_i)

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
        for x in X_test:
          y = bayes_test(x, classes, Pc, mean, cov, False)
          Y_result.append(int(y))
          y_naive = bayes_test(x, classes, Pc, mean, cov_naive, True)
          Y_result_naive.append(int(y_naive))
        
        num_err_full = 0
        num_err_naive = 0

        for y in range(len(Y_result)):
          index = Y_result[y]
          if classes[index] != Y_test[y]:
            num_err_full += 1
        for y2 in range(len(Y_result_naive)):
          index = Y_result_naive[y2]
          if classes[index] != Y_test[y2]:
            num_err_naive += 1

        print('sample, full, naive, samplenumber:', i, num_err_full, num_err_naive, len(X_test))
        err_rate_full = num_err_full
        err_rate_naive = num_err_naive

        # 5 compute difference in error rates
        diff[i] = err_rate_full - err_rate_naive
        
    print('all differences:'); print(diff)
 
    # compute mean, variance, and z-score
    mean_u = np.mean(diff)
    square_u = np.var(diff) 
    z_score = math.sqrt(K) * mean_u / square_u

    print('z-score:', z_score)

    # compute interval bound using inverse survival function of t distribution
    bound = scipy.stats.t.isf((1-alpha)/2.0, K-1) 
    print('bound:', bound)

    # output conclusion based on tests
    if -bound < z_score < bound:
        print('accept: classifiers have similar performance')
        return (bound, z_score, 'accept')
    else:
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
  for d in dataset:
     X.append(d[0:12].astype(np.float))
     Y.append(d[12])
  X = np.array(X)
  Y = np.array(Y)
  return (X, Y)

def create_D_matrix(classes, X, Y):
    """
    create classes array using X and Y
    """
    Di = []
    i=0
    for cla in classes:
      temp = []
      j = 0
      while (j < Y.size):
        if Y[j] == cla:
          temp.append(X[j])
        j += 1
      temp = np.array(temp)
      Di.append(temp)   
    Di = np.array(Di)
    return Di

def calculateProbability(x, mean, cov, isNaive = False):
  """calculate the probability density function of Fx.
  input:
  x: test case
  mean:array of means for the x's type
  cov: array of covariance matrix of x's type
  isNaive: if true, is naive bayes's operation, otherwise, full bayes

  Output: the pdf
  """
  d = len(x)
  stdev = math.fabs(np.linalg.det(cov))
  n = 1 / ((math.sqrt(2*math.pi)**d * stdev))
  temp = np.dot(x - mean[0] , np.linalg.inv(cov))
  exponentnum = math.fabs(np.dot(temp, np.transpose(x - mean[0])))
  exponent = float(math.exp(1)**(-exponentnum / 2))

  return n * exponent

#if __name__ == "__main__":
#    """
#    read in data and command-line arguments, and compute X and Y
#    """
#    if len(sys.argv) != 4:
#        print_help()
#        exit()
#    data_filename = sys.argv[1]
#    k = eval(sys.argv[2])
#    alpha = eval(sys.argv[3])
#    dataset = load_data(data_filename)
#    (X, Y) = prepare_data(dataset)
#    paired_t_test(X, Y, k, alpha)


if __name__ == "__main__":

  #dataset = load_data('D:\\Courses\\591-Programming Complex Algorithms\\Assignment2\\iris.txt')
  dataset = load_data('D:\\Courses\\591-Programming Complex Algorithms\\project\\dressing.csv')
  (X, Y) = prepare_data(dataset)
  K = 30
  alpha = 0.95
  acceptnum = 0
  times = 1
  for i in range(times):
    (bound, z_score, result)= paired_t_test(X, Y, K, alpha)
    print("[%d]: bound:%f, z_score:%f, %s" %(i, bound ,z_score ,result ))
    if result == 'accept':
      acceptnum += 1

  print('Total:')
  print("accept: %d times"  %acceptnum )
  print("reject: %d times"   %(times- acceptnum))