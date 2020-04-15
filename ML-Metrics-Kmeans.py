#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:20:30 2020

@author: kaza
"""
# module import statements
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.rcParams["figure.figsize"] = (9,6)
pd.set_option('display.max_columns', None) 
pd.options.mode.chained_assignment = None 


def narrative():
    """
    Narrative:
    The dataset is from the FBI's National Firearm's
    background check database from November 1998 to January 2019. 
    The dataset was converted from PDFs to csv format by Buzzfeed and the open source
    community and distributed on GitHub.
    
    The data has been used to view the number of denials against applications.
    "totals" are the total background check applications and "permit" are the number
    of permits issued as a result of those applications per month by state. Hand guns and
    long gun applications have been retained as columns to provide features for the dataset
    but all other applications types have been dropped. We will create some new columns as
    a step toward feeature extraction for the prediction model. 
    
    The new columns will be: 
    - "date" made by renaming the original "month" column    
    - "month" categorical, made by splitting out the months from the date column
    - "approval_rate" floats, made by dividing the values in the "permit" column by the values in "total" 
    - "approval_chance" categorical, made by binning "approval_rate" and labeling the bins
    
    Source citation: FBI Firearms National Background Check (FBI, BuzzFeed)
    Question 1: Does the dataset have missing data? Yes
    Question 2: What associated tasks can be performed with the dataset? Classification, Regression, Clustering

    """
    pass

print (narrative.__doc__)
print ("---------------------------------------------------")


################## DATA PREP  ###########################################################

url = "https://raw.githubusercontent.com/BuzzFeedNews/nics-firearm-background-checks/master/data/nics-firearm-background-checks.csv"
BGC = pd.read_csv(url) 
print ("---------------------------------------------------")
print ("Let's check the head, shape, dtypes and look for missing values on the data frame")
BGC.head(10)
# check shape
BGC.shape
# checck data types
BGC.dtypes
# confirm that there are missing values
BGC.isnull()

# Rename the "month" column to a date
print ("\n")
print ("Rename the month column to date")
BGC.rename(columns = {'month':'date'}, inplace = True) 

# Rename the "month" column to a date
print ("\n")
print ("Rename the long_gun column to rifle")
BGC.rename(columns = {'long_gun':'rifle'}, inplace = True) 
print ("---------------------------------------------------")
print ("\n")
print ("Check for missing data in the 'date' column")
print (BGC.loc[:,"date"].isnull())


def plot_elbow_graph(points):
    Error =[]
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i).fit(points)
        kmeans.fit(points)
        Error.append(kmeans.inertia_)
    plt.plot(range(1, 11), Error)
    plt.title('Elbow chart to optimize k')
    plt.xlabel('(k) No of clusters')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()
    return

def plot_cluster(points,labels,centroids):
    X = points[:,0]
    Y = points[:,1] 
    plt.scatter(X,Y, c=labels, cmap="rainbow")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='grey', s=200, alpha=0.6)
    plt.title('KMeans  (centers={}  labels={})'.format(len(centroids),len(labels))) 
    plt.grid(True)
    plt.xlabel('handgun checks')
    plt.ylabel('approval rates')
    plt.legend()
    plt.show()
    return

def show_info(column,title):
    print (column.describe())
    plt.title(title)
    plt.hist(column)
    plt.grid(True)
    plt.show()
    return

# This method will take cells from the date columns and output the month as a string
def get_month(string):
    key = "-"
    before_, key, after_ = string.partition(key)
    month_key = {"01":"January",
                 "02":"February",
                 "03":"March",
                 "04":"April",
                 "05":"May",
                 "06":"June",
                 "07":"July",
                 "08":"August",
                 "09":"September",
                 "10":"October",
                 "11":"November",
                 "12":"Decemeber"}
    
    return month_key[after_]

print ("---------------------------------------------------")
print ("Create a new categorical column called 'month' by extracting the extracting the months from the date column")
# Create an object consisting of the date column iterated through the get month method
month_categories = map(get_month,BGC.loc[:,"date"])

# Create a new categorical column for months
BGC["month"] = list(month_categories)

print ("---------------------------------------------------")
print ("Drop unused columns to lower the dimensionality of the dataset")
# Drop columns that we won't be needing that just add unecessary dimensionality to the dataset
BGC = BGC.drop(['permit_recheck', 'other', 'multiple', 'admin', 'prepawn_handgun', 'prepawn_long_gun',
                'prepawn_other', 'redemption_handgun', 'redemption_long_gun','redemption_other',
                'returned_handgun', 'returned_long_gun','returned_other', 'rentals_handgun',
                'rentals_long_gun', 'private_sale_handgun', 'private_sale_long_gun', 'private_sale_other',
                'return_to_seller_handgun', 'return_to_seller_long_gun','return_to_seller_other', ], axis=1)

# Columns after we dropped the unused 
print ("---------------------------------------------------")
print ("View the current state of the data")
print ("---------------------------------------------------")
print (BGC.columns)
BGC.shape
print ("\n")
print ("---------------------------------------------------")

print ("Show info and histograms for the remaining columns")
print ("\n")
# Show description and distribution info of "date"
show_info(BGC["date"],"BGC.date")
print ("\n")

# Show description and distribution info of "state"
show_info(BGC["state"],"BGC.state")
print ("\n")

# Show description and distribution info of "permit"
show_info(BGC["permit"],"BGC.permit")
print ("\n")

# Show description and distribution info of "handgun"
show_info(BGC["handgun"],"BGC.handgun")
print ("\n")

# Show description and distribution info of "rifle"
show_info(BGC["rifle"],"BGC.rifle")
print ("\n")

# Show description and distribution info of "rifle"
show_info(BGC["totals"],"BGC.totals")
print ("\n")

# Based on the value counts we can tell that permit has missing values
print ("Check for missing values in the 'permit' column")
pd.isnull(BGC["permit"])

# Imputing missing data background check data with means or medians would bias
# the predictions we want to make with this dataset later so we will remove rows with missing data
print ("---------------------------------------------------")
print ("Remove rows with missing data because imputing a mean or median would require splitting out the data by state and could bias future predictions")
BGC_mod = BGC.dropna()
print ("\n")
print ('The shape of the working dataset was {} and after removing rows with missing data '
       'the new shape is {}'.format(BGC.shape,BGC_mod.shape))
print ("---------------------------------------------------")
print ("\n")

# Create a new column (attribute) that represents the percentage of approved permits
approval_rate = list(BGC_mod["permit"]/BGC_mod["totals"])
BGC_mod["approval_rate"] = approval_rate

# Show description and distribution info of "approval rate"
show_info(BGC_mod["approval_rate"],"BGC_mod.approval rate")

# Show that the new column has been added to the Data frame
BGC_mod.head(10)

# Here create another column called approval chance by binning approval rate
# into 5 bins then labeling those bins categorically
BGC_mod["approval_chance"] = pd.cut(np.array(BGC_mod["approval_rate"]),
       bins=5, labels=[1, 2, 3, 4, 5])

BGC_mod.head(10)
print ("\n approval chance has been binned and turned into a categorical column")

show_info(BGC_mod["approval_chance"],"BGC_mod.approval chance")
print ("\n")
# Create a min max normalization method
def minmax(column):
    col = pd.DataFrame(column)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(col)
    return x_scaled.flatten()

def zscale(column):
    col = pd.DataFrame(column)
    z_scaled = preprocessing.StandardScaler().fit(col)
    return z_scaled.flatten()

# One Hot Encode categorical column
#months_ = pd.get_dummies(BGC_mod['month'],prefix=['month'])

# show_info(months_,"months (one hot encoded)")
#print (BGC_mod["approval_rate"])
#print (minmax(BGC_mod['approval_rate']))


print (BGC_mod.loc[:,"approval_rate"])
print (BGC_mod["approval_rate"].isnull())
apr_mean = BGC_mod["approval_rate"].mean()
# Approval rate is likely to have some NaN because of zero divisions so lets handle those
BGC_mod["approval_rate"] = BGC_mod["approval_rate"].fillna(apr_mean)



#
########## CLUSTERING #################################################################

# create a set of points from columns in the data frame for the clustering
print ("check the handgun column")
print (BGC_mod['handgun'])
print ("---------------------------------------------------")
print ("Normalize the handgun column")
print ("---------------------------------------------------")
BGC_mod['handgun'] = minmax(BGC_mod.loc[:,'handgun'])
#BGC_mod['handgun'] = zscale(BGC_mod.loc[:,'handgun'])
print ("\n")
print ("check the handgun column")
print (BGC_mod['rifle'])
print ("---------------------------------------------------")
print ("Normalize the rifle column")
BGC_mod['rifle'] = minmax(BGC_mod.loc[:,'rifle'])
#BGC_mod['rifle'] = zscale(BGC_mod.loc[:,'rifle'])
#
print ("\n")
print ("check the approval_rate column")
print (BGC_mod['approval_rate'])
print ("---------------------------------------------------")
print ("Normalize the approval_rate column")
BGC_mod['approval_rate'] = minmax(BGC_mod.loc[:,'approval_rate'])
#BGC_mod['rifle'] = zscale(BGC_mod.loc[:,'rifle'])

print ("---------------------------------------------------")
print ("Create the input for the kmeans cluster to include the normalized handgun and approval rate columns")
X = BGC_mod.loc[:,['handgun','approval_rate']].values

print ("---------------------------------------------------")
print ("We'll try the first kmeans cluster with 3 clusters")
kmeans_3 = KMeans(init='k-means++',n_clusters=3,n_init=10)
kmeans_3_labels = kmeans_3.fit_predict(X)
kmeans_3_centroids = kmeans_3.cluster_centers_
plot_cluster(X,kmeans_3_labels,kmeans_3_centroids)

print ("\n")
print ("The plot looks a little off so let's plot the elbow of the inputs to determine if there is a better number of clusters")
plot_elbow_graph(X)

print ("\n")
print ("The elbow graph suggests that 5 may be a better number of clusters")
print ("\n")
kmeans_5 = KMeans(init='k-means++',n_clusters=5,n_init=10)
kmeans_5_labels = kmeans_5.fit_predict(X)
kmeans_5_centroids = kmeans_5.cluster_centers_
plot_cluster(X,kmeans_5_labels,kmeans_5_centroids)
print ("\n")
print ("This graph with 5 clusters appears better than the versions with 3")
print ("\n")

########## MODELING #################################################################
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#
## Add Kmeeans(5 centers) to the BGC_mod dataset
#BGC_mod["kmeans_labels"] = kmeans_5_labels
##BGC_mod.head()
#
## Create a training set consisting of the permit, handgun and approval rate columns
## Can the classifier determine which group to put a pair consisting of handgun background checks
## and rifle background from a given month and state into? 
training = BGC_mod.loc[:,['handgun']].values

# On line 196 we dropped any rows with missing values so we know there should not be NaNs in training
# however the model will fail to fit if inf or very large values are present so we will impute those
#training = np.nan_to_num(training)

## Create a testing or target set consisting of the kmeans 5 center labels
def binarize_approval_chance(col):
    t = []
    for i in list(col):
        if i >=3:
            t.append(1) # high approval chance
        else:
            t.append(0) # low approval chance
    return t

print (BGC_mod.loc[:,"approval_rate"])
testing = binarize_approval_chance(BGC_mod.loc[:,"approval_chance"])
# just in case a NaN, inf or super large value slipped in
#testing = np.nan_to_num(testing)



## Split up the data into training sets and testing sets
#xTrain, xTest, yTrain, yTest = train_test_split(training, testing, test_size = 0.2, random_state = 0)
xTrain, xTest, yTrain, yTest = train_test_split(training, testing, test_size = 0.4, random_state = 2)
#



# Train a simpler Gaussian NB classifier for a comparison
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(xTrain, yTrain).predict(xTest)
print ("NaiveBayes Model accuracy: {}".format(gnb.score(xTest, yTest)))

# Train a multilayer perceptron classifier 
#    # adam is the suggested optimizer for MLP classifiers
#    # alpha is the L2 penalty - regulazation
#    # batch size is defaults to 200 but is set to 300 because of a warning

mlp = MLPClassifier(solver='lbfgs', batch_size=300, hidden_layer_sizes=(150, 10), 
                   random_state=1).fit(xTrain, yTrain)

preds = mlp.predict(xTest)

# Score the accuracy of the predictions
print ("MultiLayer Perceptron Model accuracy: {}".format(mlp.score(xTest, yTest)))
#
#
 #
#Outcome = pd.DataFrame(probabilities)
#Outcome["test"] = yTest
#Outcome["prediction"] = preds
#print (Outcome.head(10))
## Output dataframe to CSV
#Outcome.to_csv(r'KazaRazat-L08-SupervisedLearning.csv')
#
#def summary():
#    """
#    Summary:
#    After running unsupervised clustering with 5 centers on the feature set created from
#    the initial dataset it appears clear that background check permits for handguns and rifles can
#    be clustered into 5 different groups. These groups can be further translated by month and state
#    and used to determine the frequency and approval chance of gun permits for the state.
#    
#    The classification model is trained on the handgun and rifle applications. The 
#    labels are from the Kmeans clustering. An aditional feature set called approval chance (1-4) 
#    closely matches the Kmeans labels. With that feature additional predictions that map the 
#    approval chance to categories like 'very hard' or 'very easy' would be the final evolution 
#    of the project. Given a state and date what is the approval
#    chance of a handgun or rifle background check application.
#    
#    """
#    pass
#
#print (summary.__doc__)


################ MODEL METRICS #############################################################3
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

def plot_roc_curve(y,probs,model_name):
    no_model_probs = [0 for _ in range(len(y))]
    no_model_fpr, no_model_tpr, no_mode_th = roc_curve( y, no_model_probs )

    fpr, tpr, th = roc_curve( y, probs )
    
    no_model_auc = roc_auc_score( y, no_model_probs )
    model_auc = roc_auc_score( y, probs )
    
    plt.figure(figsize=(9,6))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.plot(no_model_fpr, no_model_tpr, linestyle='--', label='No model ({})'.format(no_model_auc))
    plt.plot(fpr,tpr, marker='.',label='{} ({:.2f})'.format(model_name,model_auc))
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.legend()
    plt.show()
    return

gnb_probs = gnb.predict_proba(xTest)[:,1]
mlp_probs = mlp.predict_proba(xTest)[:,1]

plot_roc_curve(yTest,gnb_probs,'Gaussian Naive Bayes')
plot_roc_curve(yTest,mlp_probs, 'Multilayer Perceptron')


# if you use the data that was labeled in the kmeans as the training data then with the clusters as the labels you should
# get a very high accuracy inside the classifications model
