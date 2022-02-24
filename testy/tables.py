import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt 
mpl.rcParams['figure.dpi'] = 150


def algo_alone(dicty ,title):
    """
    Displays the data in tables by matplotlib

    Parameters
    ----------
    dicty : List of dictionaries of MAE and MSE values for each algorithm

    Notes
    -----
    
    """
    val1 = ["mean_squared_error","mean_absolute_error"] 
    val2 = ["No deletions","LOF","EE","OneClassSVM", "IForest"]
    
    list_lists = []
    for d in dicty:
        list_lists.append(list(d.values()))
    
    val3 = list_lists

    #val3 = np.around([[0.12823345898869087, 0.32230392156862747, 0.05541549953314659, 0.3125, 0.6924019607843137],[0.0, 0.5147058823529411, 0.0857843137254902, 0.5147058823529411, 0.4852941176470588],[0.10172322138431644, 0.2867647058823529, 0.05216503267973856, 0.28431372549019607, 0.7169117647058824],[0.31747992588017293, 0.23897058823529413, 0.038101073762838465, 0.23897058823529413, 0.7610294117647058],[0.0, 0.2916666666666667, 0.048611111111111105, 0.2916666666666667, 0.7083333333333334]], decimals=3)
    val3 = np.around(val3, decimals=3)

    fig, ax = plt.subplots() 
    ax.set_axis_off() 
    table = ax.table( 
        cellText = val3,  
        rowLabels = val2,  
        colLabels = val1, 
        rowColours =["palegreen"] * 10,  
        colColours =["palegreen"] * 10, 
        cellLoc ='center',  
        loc ='upper left')         
    
    ax.set_title(title, 
                fontweight ="bold") 
    # plt.figtext(0.5, 0.52, "Accuracy percentages for each anomaly detection algorithm\n with MLPC algorithm", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":1})
    plt.show() 
    # plt.savefig("aa")
    
    

def algo_combination(dicty ,title):
    """
    Displays the data in tables by matplotlib

    Parameters
    ----------
    dicty : List of dictionaries of MAE and MSE values for each algorithm

    Notes
    -----
    
    """
    val1 = ["mean_squared_error","mean_absolute_error"] 
    val2 = ["No deletions","LOF","EE","OneClassSVM", "IForest","combination"]
    
    list_lists = []
    for d in dicty:
        list_lists.append(list(d.values()))
    
    val3 = list_lists

    #val3 = np.around([[0.12823345898869087, 0.32230392156862747, 0.05541549953314659, 0.3125, 0.6924019607843137],[0.0, 0.5147058823529411, 0.0857843137254902, 0.5147058823529411, 0.4852941176470588],[0.10172322138431644, 0.2867647058823529, 0.05216503267973856, 0.28431372549019607, 0.7169117647058824],[0.31747992588017293, 0.23897058823529413, 0.038101073762838465, 0.23897058823529413, 0.7610294117647058],[0.0, 0.2916666666666667, 0.048611111111111105, 0.2916666666666667, 0.7083333333333334]], decimals=3)
    val3 = np.around(val3, decimals=3)

    fig, ax = plt.subplots() 
    ax.set_axis_off() 
    table = ax.table( 
        cellText = val3,  
        rowLabels = val2,  
        colLabels = val1, 
        rowColours =["#1ac3f5"] * 10,  
        colColours =["#1ac3f5"] * 10, 
        cellLoc ='center',  
        loc ='upper left')         
    
    ax.set_title(title, 
                fontweight ="bold") 
    # plt.figtext(0.5, 0.52, "Accuracy percentages for each anomaly detection algorithm\n with MLPC algorithm", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":1})
    plt.show() 
    # plt.savefig("aa")
        
        
def part2(dicty ,title, val2 ):
    """
    Displays the data in tables by matplotlib

    Parameters
    ----------
    dicty : List of dictionaries of MAE and MSE values for each algorithm

    Notes
    -----
    
    """
    val1 = ["mean_squared_error","mean_absolute_error"] 
   
    
    list_lists = []
    for d in dicty:
        list_lists.append(list(d.values()))
    
    val3 = list_lists

    #val3 = np.around([[0.12823345898869087, 0.32230392156862747, 0.05541549953314659, 0.3125, 0.6924019607843137],[0.0, 0.5147058823529411, 0.0857843137254902, 0.5147058823529411, 0.4852941176470588],[0.10172322138431644, 0.2867647058823529, 0.05216503267973856, 0.28431372549019607, 0.7169117647058824],[0.31747992588017293, 0.23897058823529413, 0.038101073762838465, 0.23897058823529413, 0.7610294117647058],[0.0, 0.2916666666666667, 0.048611111111111105, 0.2916666666666667, 0.7083333333333334]], decimals=3)
    val3 = np.around(val3, decimals=3)

    fig, ax = plt.subplots() 
    ax.set_axis_off() 
    table = ax.table( 
        cellText = val3,  
        rowLabels = val2,  
        colLabels = val1, 
        rowColours =["#56b5fd"] * 10,  
        colColours =["#56b5fd"] * 10, 
        cellLoc ='center',  
        loc ='upper left')         
    
    ax.set_title(title, 
                fontweight ="bold") 
    # plt.figtext(0.5, 0.52, "Accuracy percentages for each anomaly detection algorithm\n with MLPC algorithm", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":1})
    plt.show() 
    # plt.savefig("aa")