#Code by Anthony Windmon
#email: awindmon@mail.usf.edu

#This is an extension of the work from the previous code
#Here I will use different data science & machine learning techniques for the same 'fraud detection' dataset
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#statistical significance tests
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.evaluate import mcnemar

#machine learning models
#all the models I selected have been previously used for fraud detection and were found most suitable for this type of data and its size
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #Ensemble methods
from sklearn.naive_bayes import BernoulliNB #Bayesian methods
from sklearn.tree import DecisionTreeClassifier #Decision Trees Methods

#Reading dataset
print('Gathering Data...')
transaction_dataset = pd.read_csv('C:\\Users\\awindmon\\Desktop\\transactions.csv')
transaction_df = pd.DataFrame(transaction_dataset) #Putting data into a dataframe for data manipulation
print('Shape of dataset: ',transaction_df.shape) #Number of features and samples
print('\nUnique values per field:\n', transaction_df.nunique()) #shows number of unique values per feature
print(transaction_df.info()) #shows other information about data: features with their data types, number values per feature, total amount of memory for this dataset, etc.

#Shows uniques values for specific features
#This a method that can be used to see inside of datasets that too big to open manually
for col in transaction_df:
    print('\nUnique values in', col, ':', transaction_df[col].unique())

#Manipulation of time series data
print('\nTimes series data: \n', transaction_df['transactionDateTime'])
#print times series column data type
print('\nData type is =', type(transaction_df['transactionDateTime']))
#reformatting times series using parse_dates and making time series the index using index_col
transaction_dataset = pd.read_csv('C:\\Users\\awindmon\\Desktop\\transactions.csv', parse_dates=['transactionDateTime'], index_col='transactionDateTime')
transaction_df = pd.DataFrame(transaction_dataset) #create new dataframe with new index
print('\n', transaction_df.index)
print('\nNew Data Shape =', transaction_df.shape) #prints out new shape
print(transaction_df.head)

#Figuring out which features have missing values -- in the original dataset
print('\nNumber of missing/Null values values per field (Before Removing Features):\n', transaction_df.isnull().sum())

#Removing features where there is absolutely no data
transaction_df = transaction_df.drop(columns=['echoBuffer', 'merchantCity', 'merchantState', 'merchantZip',
                                   'posOnPremises', 'recurringAuthInd'])
#Just in case there are any duplicatations that we're overlooking, we get rid of them
transaction_df = transaction_df.drop_duplicates()
#Dataset after removing empty features
print('\nNumber of missing/Null values values per field (After Removing Features):\n', transaction_df.isnull().sum())

#Considering that the remaining features only contain a few missing values, I will just drop those values
#I made this decision because all of them are missing less one percent of their data. That probably won't make much of a difference (or will it?)
transaction_df = transaction_df.dropna()
print('\nAfter dropping all missing values:\n',transaction_df.isnull().sum())
print('\nShape dataset (after removing missing values):', transaction_df.shape)

#Resampling time series data
transaction_resample = transaction_df.resample('Q').sum().plot(kind='bar') #calculates sum of features, with respect to date, per yearly quarter
#print(transaction_resample)
#plt.show()

#Label Encoding of remaining features that are 'object' and 'boolean' types
print('\nData is being encoded...')
for col in transaction_df:
    if transaction_df[col].dtype == object or bool:
        transaction_df = transaction_df.apply(LabelEncoder().fit_transform)
print('\n',transaction_df)
print('\nAfter Encoding: ',transaction_df.head)

#Finding correlation between feature 'transactionAmount' & target 'isFraud' features -- these features are highly correlated
print('\nCORRELATION BETWEEN "transactionAmount" & "isFraud":' )
transaction_corr = np.corrcoef(transaction_df['transactionAmount'],transaction_df['isFraud'])
print('Correlation: ',transaction_corr)
print('Pearson Correlation:',scipy.stats.pearsonr(transaction_df['transactionAmount'],transaction_df['isFraud']))
print('Spearman Correlation:',scipy.stats.spearmanr(transaction_df['transactionAmount'],transaction_df['isFraud']))
print('Kendalltau Correlation:',scipy.stats.kendalltau(transaction_df['transactionAmount'],transaction_df['isFraud']))

#Finding correlation between feature 'transactionType' & target 'isFraud' features -- these features are not highly correlated
print('\nCORRELATION BETWEEN "transactionType" & "isFraud":' )
transaction_corr_two = np.corrcoef(transaction_df['transactionType'],transaction_df['isFraud'])
print('Correlation: ',transaction_corr)
print('Pearson Correlation:',scipy.stats.pearsonr(transaction_df['transactionType'],transaction_df['isFraud']))
print('Spearman Correlation:',scipy.stats.spearmanr(transaction_df['transactionType'],transaction_df['isFraud']))
print('Kendalltau Correlation:',scipy.stats.kendalltau(transaction_df['transactionType'],transaction_df['isFraud']))

#Looking for outliers in data -- one way to detect outliers is by plotting data
#I only checked for a few features, but in a real world scenario I'd check them all
sn.boxplot(x=transaction_df['transactionAmount'])
#plt.show()

sn.boxplot(x=transaction_df['creditLimit'])
#plt.show()

sn.boxplot(x=transaction_df['posConditionCode'])
#plt.show()

sn.boxplot(x=transaction_df['posEntryMode'])
#plt.show()

#Determining Interquartile Range for each range
Q1 = transaction_df.quantile(0.25)
Q3 = transaction_df.quantile(0.75)
IQR = Q3 - Q1
print('\nInterquartile Range for each feature:\n',IQR)

#Wasn't a successful method
#transaction_df = transaction_df[~((transaction_df < (Q1-1.5 * IQR)) |(transaction_df > (Q3 + 1.5 * IQR))).any(axis=1)]
#print('\nShape of dataset: ',transaction_df.shape)

#Correlation Matrix -- Using Pearson Correlation (wasn't a successful method!)
'''
plt.figure(figsize=(12,10))
cor = transaction_df.corr()
sn.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#lt.show()
#Correlation with output variable
cor_target = abs(cor["isFraud"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print('\nRelevant Features = \n',relevant_features)
'''
#Separating features and target
features = transaction_df.drop(columns=['isFraud'])
target = transaction_df['isFraud']
print(features)
print(target)

#Splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=1)

#Normalizes training data using MinMaxScaler since we have multiple features
transaction_scale = MinMaxScaler()
X_trained_scaled = transaction_scale.fit_transform(X_train)

#Feature Selection
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_trained_scaled,y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print('\nTop 10 Features:\n' ,featureScores.nlargest(10,'Score'))  #print 10 best features. Highest P-Values (scores) are kept.

#10 strongest features based on SelectKBest
top_ten_features =transaction_df[['transactionAmount', 'cardPresent', 'posEntryMode', 'merchantCategoryCode',
                    'posConditionCode', 'currentBalance', 'accountNumber', 'customerId', 'cardCVV', 'merchantName']]

#5 strongest features
top_five_features = transaction_df[['transactionAmount', 'cardPresent', 'posEntryMode', 'merchantCategoryCode', 'currentBalance']]

#check if data is balanced or not (It is!)
true_count = 0
false_count = 0
for decision in target:
    if decision == 1:
        true_count+=1
    else:
        false_count+=1
print('\nCLASS INFORMATION:')
print('Amount of True labels:', true_count) #10,891 'True' labels
print('Amount of False labels:', false_count) #618,114 'False' labels

#Splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(top_ten_features, target, test_size=0.30, random_state=1) #testing set is so big because I oversampled the training set
#Splitting training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(top_ten_features, target, test_size=0.10, random_state=1)

#SMOTE to fix imbalances in data
transaction_SMOTE = SMOTE(random_state = 12, ratio = 1.0, k_neighbors = 5, sampling_strategy='minority')
X_train_smote, y_train_smote = transaction_SMOTE.fit_sample(X_train, y_train)

print('\nMODEL INFORMATION:')
print("The length of the training set =", len(X_train_smote)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample
print("The length of the validation set =", len(X_val)) #validation sample
sample_Total = len(X_train_smote) + len(X_test) + len(X_val)
print("Total number of samples =", sample_Total) # 1,346,189 samples after SMOTE (this number slightly changes on each compile)


#TRAINING ON 10 STRONGEST FEATURES
print('\nModel is training...')
print("\nScores using Top 10 Features: ")

#Random Forests
model_rf = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4)
result = model_rf.fit(X_train_smote, y_train_smote).score(X_test, y_test)
print("Random Forests Accuracy = %0.2f%%"% (result*100))

#Naive Bayes
model_nb = BernoulliNB()
result_two = model_nb.fit(X_train_smote, y_train_smote).score(X_test, y_test)
print("Naive Bayes Accuracy = %0.2f%%"% (result_two*100))

#Decison Tree
model_tree = DecisionTreeClassifier(min_samples_split=5)
result_three = model_tree.fit(X_train_smote, y_train_smote).score(X_test, y_test)
print("Decision Tree Accuracy = %0.2f%%" % (result_three*100))

#Gradient Boosting
model_gb = GradientBoostingClassifier()
result_four = model_gb.fit(X_train_smote, y_train_smote).score(X_test, y_test)
print("Gradient Boosting Accuracy = %0.2f%%" % (result_four*100))

#Splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(top_five_features, target, test_size=0.30, random_state=1) #testing set is so big because I oversampled the training set
#Splitting training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(top_five_features, target, test_size=0.10, random_state=1)

#SMOTE to fix imbalances in data
transaction_SMOTE = SMOTE(random_state = 12, ratio = 1.0, k_neighbors = 5, sampling_strategy='minority')
X_train_smote, y_train_smote = transaction_SMOTE.fit_sample(X_train, y_train)

#TRAINING ON 5 STRONGEST FEATURES
print("\nScores using Top 5 Features: ")

#Random Forests
model_rf = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4)
model_rf.fit(X_train_smote, y_train_smote) #Training model
result = model_rf.score(X_test, y_test)
print("Random Forests Accuracy = %0.2f%%"% (result*100))

#Naive Bayes
model_nb = BernoulliNB()
model_nb.fit(X_train_smote, y_train_smote)
result_two = model_nb.score(X_test, y_test)
print("Naive Bayes Accuracy = %0.2f%%"% (result_two*100))

#Decison Tree
model_tree = DecisionTreeClassifier(min_samples_split=5)
model_tree.fit(X_train_smote, y_train_smote)
result_three = model_tree.score(X_test, y_test)
print("Decision Tree Accuracy = %0.2f%%" % (result_three*100))

#Gradient Boosting
model_gb = GradientBoostingClassifier()
model_gb.fit(X_train_smote, y_train_smote)
result_four = model_gb.score(X_test, y_test)
print("Gradient Boosting Accuracy = %0.2f%%" % (result_four*100))

#To determine the final model for this problem, I will a 5x2 Cross Validation paired t test
#This test is used to determine statistical difference between two classifiers
#Citation: http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/
print('\nResults of 5x2 Cross Validation Paired T-Test: ')
t, p = paired_ttest_5x2cv(estimator1=model_rf, estimator2=model_nb, X=X_train_smote, y=y_train_smote, random_seed=1)
print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

if (p>t):
    print('5x2 CV: The null hypothesis is not rejected. There is no statistical difference between classifiers.')
else:
    print('5x2 CV: The null hypothese is rejected. There is a statistical difference between classifiers.')
