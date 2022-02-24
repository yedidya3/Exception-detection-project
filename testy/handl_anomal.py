import numpy as np

'''
Auxiliary functions
'''




def lower_idc(X_train,num):
    """
    Returns the indexes with the lowest values in the array

    Parameters
    ----------
    X_train : list

    num : The number of values that will be repeated
    
    Return : List of low value indexes 
    
    Notes
    -----
    """
    np.argsort(X_train)[:num]
    
def delete_outliers(X_train,y_train,list_idc):
    """
    Deleting a list of indexes from an array
    
    Parameters
    ----------
    X_train : list

    y_train : list
    
    list_idc : list of idc
    
    Return : X_train y_train Without the values of the indexes
    
    Notes
    -----
    """
    for index in sorted(list_idc, reverse=True):
        del X_train[index]
        del y_train[index]
        return X_train,y_train


def count_minus(listy):
    """
    Counting values lower than 0 in the array
    
    Parameters
    ----------
    listy : list

   
    Return : The number of values is less than 0
    
    Notes
    -----
    """
    neg_count = 0
    for num in listy:
          
        # checking condition
        if num < 0:
            neg_count += 1
            
    return neg_count
          