
#regression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from sklearn.preprocessing import label_binarize

def acc(Y , Y_hat):
    """
    Measuring the amount of accuracy in the data classification

    Parameters
    ----------
    Y : The right results

    Y_hat :Estimated results
    
    Return : Percentage accuracy
    
    Notes
    -----
    """
    i = 0
    for y,y_hat in zip(Y,Y_hat):
        if y==y_hat:
            i+=1
    return(i/len(Y))


def metrics_regression(y , y_hat):
    """
    Measuring the amount of accuracy in the data Regression

    Parameters
    ----------
    y : The right results

    y_hat :Estimated results
    
    Return : Dictionary of accuracy results
    
    Notes
    -----
    """
    diction = {}
    diction["mean_squared_error"] = mean_squared_error(y , y_hat)
    diction["mean_absolute_error"] = mean_absolute_error(y , y_hat)
    return diction
  
  
def metrics_classification_multi(y , y_hat,y_pred,labelss):
    """
    Measuring the amount of accuracy in the data Multi classification

    Parameters
    ----------
    y : The right results

    y_hat :Estimated results
    
    labelss - Tags of labeling
    
    Return : Dictionary of accuracy results
    
    Notes
    -----
    """
    
    diction = {}
 
    diction["roc_auc_score"] = roc_auc_score(y , y_pred,multi_class='ovr')

    diction["f1_score"] = f1_score(y , y_hat,average='weighted')
   
    return diction
    # None, 'micro', 'macro', 'weighted'
    
    
def metrics_classification_binary(y , y_hat):
    """
    Measuring the amount of accuracy in the data binary classification

    Parameters
    ----------
    y : The right results

    y_hat :Estimated results
    
    Return : listDictionary of accuracy results
    
    Notes
    -----
    """
    diction = {}
    diction["Accuracy_p"] = acc(y , y_hat)
    diction["f1_score"] = f1_score(y , y_hat)
    diction["roc_auc_score"] = roc_auc_score(y , y_hat)
    return diction
    

