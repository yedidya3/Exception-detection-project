
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import precision_m
import pandas as pd
import numpy as np
from itertools import chain, combinations

def improvement_test_regression(df,algos):
    """
    Accuracy test on Data
    Running the algorithms to detect anomalies on data
    Deleting the anomalies
    Predicting the values
    Accuracy test

    Parameters
    ----------
    df : Data Frame
    
    algos: list of algorithms to detect anomalies
    
    return : list of Dictionaris of accuracy values
    
    Notes
    -----
    
    """    
    data = df.values
  
    # split into input and output elements
    X, Y = data[:, :-1], data[:, -1]   

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) #train_test_split(X, Y, test_size=0.25, random_state=42)
    
    accuracy=[]
    clf = LinearRegression()#MLPRegressor(random_state=1, max_iter=500)# LinearRegression()
    clf.fit(X_train,y_train)
    y_hat = clf.predict(X_test)
    accuracy.append(precision_m.metrics_regression(y_test,y_hat))
    
    #Measurement of each algorithm
    for algo in algos:
        # algo.fit(X_train)
        outliers = algo.fit_predict(X_train)
        mask = outliers != -1
      
        X_train_neto, y_train_neto = X_train[mask, :], y_train[mask]
        clf_neto = LinearRegression()
        clf_neto.fit(X_train_neto,y_train_neto)
        y_hat_neto=clf_neto.predict(X_test)
        accuracy.append(precision_m.metrics_regression(y_test,y_hat_neto))
     
    return accuracy
    # select all rows that are not outliers
    

def clean_dataset(df):
    """
    clean data set from values nan , inf, -inf

    Parameters
    ----------
    df : Data Frame
     
    return: Data Frame
    
    Notes
    -----
    
    """      
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def delete_all(df,algos):
    """
    Running any anomaly detection algorithms
    Mark any anomalies that appear
    Delete the anomalies and run an identification algorithm

    Parameters
    ----------
    df : Data Frame
    
    algos: list of algorithms to detect anomalies
    
    return : list of Dictionaris of accuracy values
    
    Notes
    -----
    
    """  
    data = df.values
    X, Y = data[:, :-1], data[:, -1]   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) 
    list_lists_outliers = []
    for algo in algos:
        # algo.fit(X_train)
        list_lists_outliers.append(algo.fit_predict(X_train))
        
    lst_1 = np.array([1] * (len(list_lists_outliers[0])))
    
    for list_outliers in list_lists_outliers:
        for i in range(len(list_outliers)):
            if list_outliers[i] == -1:
               lst_1[i]= -1 
    mask = lst_1 != -1

    X_train_neto, y_train_neto = X_train[mask, :], y_train[mask]
    clf_neto = LinearRegression()
    clf_neto.fit(X_train_neto,y_train_neto)
    y_hat_neto=clf_neto.predict(X_test)
    accuracy = precision_m.metrics_regression(y_test,y_hat_neto)  
    return accuracy
    # select all rows that are not outlie

def Add_and_delete(df,algos):
    """
    Connecting all the abnormal lists of all the algorithms
    Marking the lowest value as an exception
    Deleting and running linearRegreesion
    
    Parameters
    ----------
    df : Data Frame
    
    algos: list of algorithms to detect anomalies
    
    return : list of Dictionaris of accuracy values
    
    Notes
    -----
    
    """  
    data = df.values
    X, Y = data[:, :-1], data[:, -1]   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) #train_test_split(X, Y, test_size=0.25, random_state=42)

    list_lists_outliers = []
    for algo in algos:
        # algo.fit(X_train)
        list_lists_outliers.append(algo.fit_predict(X_train))
        
    lst_1 = np.array([0] * (len(list_lists_outliers[0])))

    for list_outliers in list_lists_outliers:
        lst_1+=list_outliers
    minimal_value = np.amin(lst_1)
    mask = lst_1 != minimal_value

    X_train_neto, y_train_neto = X_train[mask, :], y_train[mask]
    clf_neto = LinearRegression()
    clf_neto.fit(X_train_neto,y_train_neto)
    y_hat_neto=clf_neto.predict(X_test)
    accuracy = precision_m.metrics_regression(y_test,y_hat_neto)  
    return accuracy

def delete_multiple_element(list_object, indices):
    """
    Delete indexes from a list
    
    Parameters
    ----------
    df : list_object
    
    indices: list of indices
    
    return : list without indices values
    
    Notes
    -----
    
    """  
    
    list_object = list(list_object)
    indices = sorted(indices, reverse=True)
    i= 0
    for idx in indices: 
        if idx < len(list_object):
            list_object.pop(idx)
    return np.array(list_object)   
    
    
def uniformy_average(df,algos):
    """
    use in decision_function : Average anomaly score of X of the base classifiers.
    An equal ratio is given to each algorithm connected and deleted 10 percent of the data
    
    Parameters
    ----------
    df : data frame
    
    algos: list of abnormal detection algorithms 
    
    return : list of Dictionaris of accuracy values
    
    Notes
    -----
    
    """
    data = df.values
    X, Y = data[:, :-1], data[:, -1]   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) #train_test_split(X, Y, test_size=0.25, random_state=42)
    
    list_lists_outliers = []
    for algo in algos:
        algo.fit(X_train)
        list_lists_outliers.append(algo.decision_function(X_train))
    list_lists_outliers = np.array(list_lists_outliers)
    avarage = list_lists_outliers.mean(0)
    args_min = (np.argsort(avarage))[:int(len(avarage)/15)]
    # print(avarage[134])
    
    X_train_neto, y_train_neto = X_train, y_train
    X_train_neto = delete_multiple_element(X_train_neto,args_min)
    y_train_neto = delete_multiple_element(y_train_neto,args_min)
    
    clf_neto = LinearRegression()
    clf_neto.fit(X_train_neto,y_train_neto)
    y_hat_neto=clf_neto.predict(X_test)
    accuracy = precision_m.metrics_regression(y_test,y_hat_neto)  
    return accuracy





def ratio_average(df,algos,ratio,diff):
    """
    use in decision_function : Average anomaly score of X of the base classifiers.
    An ratio is given to each algorithm connected and deleted 10 percent of the data
    
    Parameters
    ----------
    df : data frame
    
    algos: list of abnormal detection algorithms 
    
    ratio: Ratio list
    
    Return 
    ----------
    list of Dictionaris of accuracy values
    
    Notes
    -----
    
    """
    data = df.values
    X, Y = data[:, :-1], data[:, -1]   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) #train_test_split(X, Y, test_size=0.25, random_state=42)
    
    list_lists_outliers = []
    for algo,r in zip(algos,ratio):
        algo.fit(X_train)
        list_lists_outliers.append(r*algo.decision_function(X_train))
    list_lists_outliers = np.array(list_lists_outliers)
    lst_1 = np.array([0] * (len(list_lists_outliers[0])))

    for list_outliers in list_lists_outliers:
        lst_1 = np.sum([lst_1, np.array(list_outliers)], axis=0)
      
    
    args_min = (np.argsort(lst_1))[:int(len(lst_1)/diff)]
    # print(avarage[134])
    
    X_train_neto, y_train_neto = X_train, y_train
    X_train_neto = delete_multiple_element(X_train_neto,args_min)
    y_train_neto = delete_multiple_element(y_train_neto,args_min)
   
    
    clf_neto = LinearRegression()
    clf_neto.fit(X_train_neto,y_train_neto)
    y_hat_neto=clf_neto.predict(X_test)
    accuracy = precision_m.metrics_regression(y_test,y_hat_neto)  
    return accuracy


def all_subsets(ss):
    """
    Create all subgroups in the range 
    
    Parameters
    ----------
    ss : list
    
    Return 
    ----------
    list touples of subgroups of accuracy values
    
    Notes
    -----
    
    """
    ret =list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))
    del ret[0]
    return ret
    


def combination_delete_all(indc,list_lists_outliers,X_train,y_train,X_test,y_test):
    """
  
    Calculate the accuracy of the algorithm after deleting all the anomalies
    
    Parameters
    ----------
    indc : list of indc
    list_lists_outliers :
    X_train :
    y_train :
    X_test :
    y_test :
    
    
    Return 
    ----------
    accuracy : dictionary of accurancy 
    lst_1 : ,list of marking anomalies
    Notes
    -----
    
    """
    lst_1 = np.array([1] * (len(list_lists_outliers[0])))

    for index in indc:
        list_outliers = list_lists_outliers[index]
        for i in range(len(list_outliers)):
            if list_outliers[i] == -1:
                lst_1[i]= -1 
    mask = lst_1 != -1

    X_train_neto, y_train_neto = X_train[mask, :], y_train[mask]
    clf_neto = LinearRegression()
    clf_neto.fit(X_train_neto,y_train_neto)
    y_hat_neto=clf_neto.predict(X_test)
    accuracy = precision_m.metrics_regression(y_test,y_hat_neto)  
    return accuracy,lst_1

def combination_Add_and_delete(indc,list_lists_outliers,X_train,y_train,X_test,y_test):
    """
    
    Calculate the accuracy of the algorithm after deleting all the anomalies
    
    Parameters
    ----------
    indc : list of indc
    list_lists_outliers :
    X_train :
    y_train :
    X_test :
    y_test :
    
    
    Return 
    ----------
    accuracy : dictionary of accurancy 
    lst_1 : ,list of marking anomalies
    Notes
    -----
    
    """ 
    lst_1 = np.array([0] * (len(list_lists_outliers[0])))

    for index in indc:
        list_outliers = list_lists_outliers[index]
        lst_1+=list_outliers
    minimal_value = np.amin(lst_1)
    mask = lst_1 != minimal_value

    X_train_neto, y_train_neto = X_train[mask, :], y_train[mask]
    clf_neto = LinearRegression()
    clf_neto.fit(X_train_neto,y_train_neto)
    y_hat_neto=clf_neto.predict(X_test)
    accuracy = precision_m.metrics_regression(y_test,y_hat_neto) 
    
    
    lst_2 = np.array([1] * (len(list_lists_outliers[0])))
    for j in  range(len(lst_1)):
        if lst_1[j] ==minimal_value:
            lst_2[j] =-1
            
    return accuracy,lst_2

def combination(df,algos,result_best):
    """
    Go through all the subgroups of the exception lists
    Calculate the accuracy of the algorithm after deleting all the anomalies
    
    Parameters
    ----------
    indc : list of indc
    list_lists_outliers :
    X_train :
    y_train :
    X_test :
    y_test :
    
    
    Return 
    ----------
    accuracy : dictionary of accurancy 
    lst_2 : ,list of marking anomalies
    Notes
    -----
    
    """
    data = df.values
    X, Y = data[:, :-1], data[:, -1]   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) #train_test_split(X, Y, test_size=0.25, random_state=42) 
   
    list_lists_outliers = []
    for algo in algos:
        # algo.fit(X_train)
        list_lists_outliers.append(algo.fit_predict(X_train))
    
    indc = []
    
    
    combinatio_idc = all_subsets(list(range(0,3)))
    mse=[]
    mae=[]
    for idc in combinatio_idc:        
        results_temp,lst_outliers = combination_delete_all(idc,list_lists_outliers,X_train,y_train,X_test,y_test)
        if result_best['mean_squared_error']>results_temp['mean_squared_error']:
            result_best['mean_squared_error']=results_temp['mean_squared_error']
            mse=lst_outliers
        if result_best['mean_absolute_error']>results_temp['mean_absolute_error']:
            result_best['mean_absolute_error']=results_temp['mean_absolute_error']
            mae=lst_outliers    
        results_temp,lst_outliers = combination_Add_and_delete(idc,list_lists_outliers,X_train,y_train,X_test,y_test)
        if result_best['mean_squared_error']>results_temp['mean_squared_error']:
            result_best['mean_squared_error']=results_temp['mean_squared_error']
            mse=lst_outliers
        if result_best['mean_absolute_error']>results_temp['mean_absolute_error']:
            result_best['mean_absolute_error']=results_temp['mean_absolute_error']
            mae=lst_outliers
    
    return {'mean_squared_error':mse,'mean_absolute_error':mae,"result_best":result_best}



 