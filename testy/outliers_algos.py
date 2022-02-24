from scipy import stats
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import collections


class outlier_algo:
    def __init__(self,model):
        

        self.model=model

    def fit(self, X_train):
        
        self.model.fit(X_train)
    def fit(self, X_train,y_train):
        self.model.fit(X_train)
    def predict_outliers(self , X_test):     
        decF = self.model.decision_function(X_test)
                
        neg_count = 0
        # iterating each number in list
        for num in decF:
            # checking condition
            if num < 0:
                neg_count += 1
  
        #decF_ee = ee.decision_function(X_train4)
        decF_lofZ = stats.zscore(decF)
        
        return decF_lofZ,neg_count 
    def fit_predict(self,X_train):
        return self.model.fit_predict(X_train)
    
    
class LOF_Outliers(outlier_algo):
    def __init__(self):
        outlier_algo.__init__(self,LocalOutlierFactor())
        
class EllipticEnvelopeOutliers(outlier_algo):
    def __init__(self):
        outlier_algo.__init__(self,EllipticEnvelope())

class OneClassSVMOutliers(outlier_algo):
    def __init__(self):
        outlier_algo.__init__(self,OneClassSVM(gamma="auto"))

class IforestOutliers(outlier_algo):
    def __init__(self):
        outlier_algo.__init__(self,IsolationForest(n_estimators = 1000))

class DecisionTreeOutliers(outlier_algo):
    def __init__(self,num_of_leafs):
        self.num_of_leafs = num_of_leafs
        outlier_algo.__init__(self,DecisionTreeClassifier(max_leaf_nodes=self.num_of_leafs, random_state=1))
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
           
        ss = self.model.apply(X_train)

        occurrences = dict(collections.Counter(ss))
        for i in occurrences:
            occurrences[i] = float(occurrences[i]/len(ss))
        def zify_scipy(d):
            keys, vals = zip(*d.items())
            return dict(zip(keys, stats.zscore(vals, ddof=1)))
        self.occurrences = zify_scipy(occurrences)  
        
                
    def predict_outliers(self , X_train):     
        ss = self.model.apply(X_train)
        exceptional_probability = []
        for s in ss:
            exceptional_probability.append(self.occurrences[s])
        #need change
        return exceptional_probability,int(len(X_train)/10)
    
    def predict(self ,X_test, check_input=True):
        return self.model.predict(X_test)
    
    def fit_predict(self,X_train):
        ss = self.model.apply(X_train)
        exceptional_probability = []
        for s in ss:
            exceptional_probability.append(self.occurrences[s])
       
        order2 = list(np.argsort(exceptional_probability))[:int(len(X_train)/10)]
        ret = []
        for i in range(len(X_train)):
            if i in order2:
                ret.append(-1)
            else:
                ret.append(1)
        #need change
        return ret
                
                
     
        
        


        
        
        


        
        
        
