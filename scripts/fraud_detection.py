#Code by Anthony Windmon
#email: awindmon@mail.usf.edu 

import json, csv, numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import collections

#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#The data is arranged using a JSON, dictionary data structure, with a key:value setup.
#Each key is a feature name (i.e., accountNumber, customerId, creditLimit, etc), and the values are the 'raw data'.
#There are 641,914 records (overall) and 29 fields

#Convert transactions JSON file to a csv file for better assessment.
print('Working...')
transaction_data = []
for data in open('transactions\\transactions.txt', 'r'): #you may have to change this path, with respect to your own system
    transaction_data.append(json.loads(data))
df = pd.DataFrame(transaction_data)
transaction_csv = df.to_csv('transactions.csv', index=False)

#Reading shape of dataset
dataset = 'C:\\Users\\awindmon\\Desktop\\transactions.csv' #you may have to change this path, with respect to your own system
raw_data = open(dataset, 'r')
reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
list_name = list(reader)
data = numpy.array(list_name)
shape = data.shape
print('Shape of data: ', data.shape) #prints out number of samples, and number of features

#Number of null/missing, min/max and unique values in each field
transaction_df = pd.read_csv('C:\\Users\\awindmon\\Desktop\\transactions.csv') #you may have to change this path, with respect to your own system
print('\nNumber of Null values per field: \n', transaction_df.isnull().sum())
print('\nMinimum value per field:\n', transaction_df.min())
print('\nMaximum value per field:\n', transaction_df.max())
print('\nUnique values per field:\n', transaction_df.nunique())
print('\nAdditional stats about data:\n',transaction_df.describe())

#Histogram of transactionAmount column
plt.hist(transaction_df['transactionAmount'])
plt.title('Histogram of Transaction Amount (transactionAmount)')
plt.xlabel('Amount ($) Spent per Transaction')
plt.ylabel('Number of Transactions (Overall)')
#plt.show() #Uncomment if you'd like to see this figure!

#Identifies the number of transactions for each transaction type
reversed_transcations = []
purchase_transactions = []
verification_transactions = []
nan_transactions = []
reversed_transcations_amount = 0
for trans_type in transaction_df['transactionType']:
    if trans_type == 'REVERSAL':
        reversed_transcations.append(trans_type)
    if trans_type == 'PURCHASE':
        purchase_transactions.append(trans_type)
    if trans_type == 'ADDRESS_VERIFICATION':
        verification_transactions.append(trans_type)
    if trans_type == 'nan':
        nan_transactions.append(trans_type)

print('TRANSACTION INFORMATION:')
print('Total amount of reversed transactions: ',reversed_transcations.count('REVERSAL'))
print('Total amount of purchases: ',purchase_transactions.count('PURCHASE'))
print('Total amount of verification transactions: ',verification_transactions.count('ADDRESS_VERIFICATION'))
print('Total amount of NaN transactions: ',nan_transactions.count('nan'))

#Creating dictionary with transactionType and transactionAmount, to identify transaction amount for each transaction type
#Citation: EdChum - Reinstate Monica (2015), "Pandas - Convert two columns into a new column as a dictionary", [Source code]. https://stackoverflow.com/questions/33378731/pandas-convert-two-columns-into-a-new-column-as-a-dictionary
transaction_df['associatedSum'] = transaction_df.apply(lambda row: {row['transactionType']:row['transactionAmount']}, axis=1)
#Citation: Geeksforgeeks (2020), "Python | Sum list of dictionaries with same key", [Source code]. https://www.geeksforgeeks.org/python-sum-list-of-dictionaries-with-same-key/
associatedSum_counter = collections.Counter()
for sum in transaction_df['associatedSum']:
    associatedSum_counter.update(sum)
associatedSum_result = dict(associatedSum_counter) #keeps count transaction amount for each transaction type
print('Transaction amount ($) per field: ',associatedSum_result)
print('Total Transaction Amount ($) Overall: %0.2f'% (transaction_df['transactionAmount'].sum()))
#print(transaction_df['associatedSum']) #prints out 'associatedSum' dictionary

#Setting up data before modelling
print('\nCATEGORICAL FEATURE INFORMATION (BEFORE ENCODING):')
print('Country: ',transaction_df['acqCountry'].unique())
print('Card Present: ',transaction_df['cardPresent'].unique())
print('Expiration Date: ',transaction_df['expirationDateKeyInMatch'].unique())
print('Merchant Category: ',transaction_df['merchantCategoryCode'].unique())
print('Merchant Country Code: ',transaction_df['merchantCountryCode'].unique())
print('Transaction Type: ',transaction_df['transactionType'].unique())
print('Fraud: ',transaction_df['isFraud'].unique())

#removing features with absolutely no data from our list of features to be considered
transaction_df = transaction_df.drop(columns=['echoBuffer', 'merchantCity', 'merchantState', 'merchantZip',
                                   'posOnPremises', 'recurringAuthInd', 'transactionDateTime',
                                   'accountOpenDate', 'dateOfLastAddressChange', 'currentExpDate', 'associatedSum'])

#filling in Nan/Missing values with mode values
print('\nFilling in missing/Nan values...')
pd.DataFrame(transaction_df).fillna(transaction_df.mode())
transaction_data = pd.DataFrame(transaction_df)
#Citation: LeDoux, J. (2019), "Impute Missing Values", [Source code]. https://jamesrledoux.com/code/imputation
filled_transaction_df = transaction_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
print('\nAmount of missing/Nan values after filling: \n',filled_transaction_df.isnull().sum())

#Label Encoder - changing catergorical data ---> numerical data, so our classifier can read it
transaction_encoder = LabelEncoder()
filled_transaction_df['acqCountry'] = transaction_encoder.fit_transform(filled_transaction_df['acqCountry'])
filled_transaction_df['cardPresent'] = transaction_encoder.fit_transform(filled_transaction_df['cardPresent'])
filled_transaction_df['expirationDateKeyInMatch'] = transaction_encoder.fit_transform(filled_transaction_df['expirationDateKeyInMatch'])
filled_transaction_df['merchantCategoryCode'] = transaction_encoder.fit_transform(filled_transaction_df['merchantCategoryCode'])
filled_transaction_df['merchantCountryCode'] = transaction_encoder.fit_transform(filled_transaction_df['merchantCountryCode'])
filled_transaction_df['transactionType'] = transaction_encoder.fit_transform(filled_transaction_df['transactionType'])
filled_transaction_df['merchantName'] = transaction_encoder.fit_transform(filled_transaction_df['merchantName'])
filled_transaction_df['isFraud'] = transaction_encoder.fit_transform(filled_transaction_df['isFraud'])
#print(filled_transaction_df)

#Double check to make sure all features are encoded
print('\nCATEGORICAL FEATURE INFORMATION (AFTER ENCODING):')
print('Country: ',filled_transaction_df['acqCountry'].unique())
print('Card Present: ',filled_transaction_df['cardPresent'].unique())
print('Expiration Date: ',filled_transaction_df['expirationDateKeyInMatch'].unique())
print('Merchant Category: ',filled_transaction_df['merchantCategoryCode'].unique())
print('Merchant Country Code: ',filled_transaction_df['merchantCountryCode'].unique())
print('Transaction Type: ',filled_transaction_df['transactionType'].unique())
print('Merchant Name: ',filled_transaction_df['merchantName'].unique())
print('Fraud: ',filled_transaction_df['isFraud'].unique())

#target is 'isFraud'
target = filled_transaction_df['isFraud']

#check if data is balanced or not (It is!)
true_count = 0
false_count = 0
for decision in filled_transaction_df['isFraud']:
    if decision == 1:
        true_count+=1
    else:
        false_count+=1
print('\nCLASS INFORMATION:')
print('Amount of True labels:', true_count) #11,302 'True' labels
print('Amount of False labels:', false_count) #630,612 'False' labels


#Splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(filled_transaction_df.drop(columns=['isFraud']), target, test_size=0.20,
                                                        random_state=42)
print('\nMODEL INFORMATION:')
print("The length of the training set =", len(X_train)) #training sample
print("The length of the testing set =", len(X_test)) #testing sample

print('\nModel is being trained...')

#Random Forest Classifier
model = RandomForestClassifier(n_estimators=150,random_state=0, max_depth=15,min_samples_leaf=2,
                                min_samples_split=4)
model.fit(X_train, y_train) #Training model

#Feature Selection
model_features = ExtraTreesClassifier(n_estimators=100)
model_features.fit(X_train, y_train)
print('\nFeature Scores (Using feature_importances_ function): ',model_features.feature_importances_) #feature_importances_ is a feature which gives us the strongest features
strongest_features = pd.Series(model_features.feature_importances_, index=X_train.columns)
strongest_features.nlargest(10).plot(kind='barh')
plt.xlabel('Scores')
plt.ylabel('Features')
plt.title('10 Strongest Features')
#plt.show() #Uncomment if you'd like to see this figure!

print('\n--------------RANDOM FORESTS----------------')

result = model.score(X_test, y_test)
print("Random Forests Accuracy = %0.2f"% (result*100))

#Confusion Matrix
y_predicted = model.predict(X_test)
confuse_matrix = confusion_matrix(y_test, y_predicted)
print("Random Forests Confusion Matrix: \n", confuse_matrix)

#plot confusion matrix for Random Forests classifier
fig = plt.figure()
csfont = {'fontname':'Times New Roman'}
plt.rcParams['font.family'] = 'Times New Roman'
sn.heatmap(confuse_matrix, fmt="d", annot=True, cbar=False,annot_kws={"size": 20})
ax = fig.add_subplot(1,1,1)
plt.title("Fraud Detection Confusion Matrix -- RF Classifier", fontsize=20,**csfont)
plt.xlabel('Predicted Label', fontsize=20,**csfont)
plt.ylabel('Truth Label', fontsize=20, **csfont)
ax.xaxis.set_ticklabels(['FALSE', 'TRUE'], fontsize=20, horizontalalignment ='center')
ax.yaxis.set_ticklabels(['FALSE', 'TRUE'], fontsize=20, verticalalignment='center')
#plt.show() #Uncomment if you'd like to see this figure!

#10 fold cross validation - THIS TAKES A LONG TIME TO CALCULATE!
#print('Calculating 10-Fold Cross Validation (takes a minute)...')
#cv_scores = cross_val_score(model, filled_transaction_df.drop(columns=['isFraud']), target, cv=10)
#print("10-Fold scores = ", cv_scores)
#print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))

#precision, recall & f-measure
classifier_report_rf = classification_report(y_test,y_predicted)
print(classifier_report_rf)

print('\n--------------LOGISTIC REGRESSION----------------')

#Logistic Regression Classifier
model_lr = LogisticRegression(max_iter=150, solver='lbfgs')
model_lr.fit(X_train, y_train)
scores = model_lr.score(X_test,y_test)
print("Logistic Regression = %0.2f"% (scores*100))

#Confusion Matrix
y_predicted = model_lr.predict(X_test)
confuse_matrix = confusion_matrix(y_test, y_predicted)
print("Logistic Regression Confusion Matrix: \n", confuse_matrix)

#plot confusion matrix for Logistic Regression classifier
fig = plt.figure()
csfont = {'fontname':'Times New Roman'}
plt.rcParams['font.family'] = 'Times New Roman'
sn.heatmap(confuse_matrix, fmt="d", annot=True, cbar=False,annot_kws={"size": 20})
ax = fig.add_subplot(1,1,1)
plt.title("Fraud Detection Confusion Matrix -- LR Classifier", fontsize=20,**csfont)
plt.xlabel('Predicted Label', fontsize=20,**csfont)
plt.ylabel('Truth Label', fontsize=20, **csfont)
ax.xaxis.set_ticklabels(['FALSE', 'TRUE'], fontsize=20, horizontalalignment ='center')
ax.yaxis.set_ticklabels(['FALSE', 'TRUE'], fontsize=20, verticalalignment='center')
#plt.show() #Uncomment if you'd like to see this figure!

#10 fold cross validation
cv_scores = cross_val_score(model_lr, filled_transaction_df.drop(columns=['isFraud']), target, cv=10)
print("10-Fold scores = ", cv_scores)
print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))

#precision, recall & f-measure
classifier_report_lr = classification_report(y_test,y_predicted)
print(classifier_report_lr)

print('\n--------------NAIVE BAYES----------------')

#Naive Bayes Classifier
model_NB = GaussianNB()
model_NB.fit(X_train, y_train)
result_NB = model_NB.score(X_test, y_test)
print("Naive Bayes = %0.2f"% (scores*100))

#Confusion Matrix
y_predicted = model_NB.predict(X_test)
confuse_matrix = confusion_matrix(y_test, y_predicted)
print("Naive Bayes Confusion Matrix: \n", confuse_matrix)

#plot confusion matrix for Naive Bayes classifier
fig = plt.figure()
csfont = {'fontname':'Times New Roman'}
plt.rcParams['font.family'] = 'Times New Roman'
sn.heatmap(confuse_matrix, fmt="d", annot=True, cbar=False,annot_kws={"size": 20})
ax = fig.add_subplot(1,1,1)
plt.title("Fraud Detection Confusion Matrix -- NB Classifier", fontsize=20,**csfont)
plt.xlabel('Predicted Label', fontsize=20,**csfont)
plt.ylabel('Truth Label', fontsize=20, **csfont)
ax.xaxis.set_ticklabels(['FALSE', 'TRUE'], fontsize=20, horizontalalignment ='center')
ax.yaxis.set_ticklabels(['FALSE', 'TRUE'], fontsize=20, verticalalignment='center')
#plt.show() #Uncomment if you'd like to see this figure!

#10 fold cross validation
cv_scores = cross_val_score(model_NB, filled_transaction_df.drop(columns=['isFraud']), target, cv=10)
print("10-Fold scores = ", cv_scores)
print("Avg. Accuracy (of 10-FCV) = %0.2f (+/- %0.2f)"% (cv_scores.mean(), cv_scores.std()*2))

#precision, recall & f-measure
classifier_report_NB = classification_report(y_test,y_predicted)
print(classifier_report_NB)

