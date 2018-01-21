# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:38:52 2018

@author: DELL
"""

#Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import re
import seaborn as sns
sns.set()

#Loading the dataset
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Basic information fetching
print train.info()
print train.shape
print train.head(10)
print train.columns


#finding the columns with the null values
train_null=train.isnull().sum()
test_null=test.isnull().sum()
print "Null values in train data:--->"
print train_null
print "\n"
print "Null values in test data :---->"
print test_null


#********************Performed feature engineering******************
#Visualising the features with the target
def plot(feature):
    approved=train[train.Approved==1][feature].value_counts()
    not_approved=train[train.Approved==0][feature].value_counts()
    df=pd.DataFrame({'approved':approved, 'not_approved':not_approved})
    df.plot.bar()
    plt.show()
    
    
plot('Gender')
plot('City_Code')
plot('City_Category')#there are three city category A, B, C
plot('Employer_Category1') #3 Category A, B, C
plot('Employer_Category2') #4 category 1, 2, 3, 4
plot('Customer_Existing_Primary_Bank_Code')
plot('Primary_Bank_Type') #two types P, G
plot('Contacted') # Y, N
plot('Source_Category') #A, B, C, D, E, F, G
plot('Loan_Period')
plot('Loan_Amount')
plot('Var1') #0, 2, 4, 7, 10

#combining the train and test data into a list
train_test_dataset=[train, test]

#*******************************Gender************************
#Mapping the Gender
#male:0
#female:1

gender_mapping={'Male':0, 'Female':1}
for dataset in train_test_dataset:
    dataset['Gender']=dataset['Gender'].map(gender_mapping)

print train.Gender.head(10)
print test.Gender.head(10)    

#*******************************DOB******************************

#*******************************TEST***********************************
dob_train=train['DOB'].astype(str).tolist()
#print dob
new_list_train=[]
for i in dob_train:
    
    result=re.findall(r'\d{2}/\d{2}/(\d{2})', i)
    new_list_train.append(result)

#print new_list

year_list_train=[]
for i in new_list_train:
    for j in i:
      year_list_train.append('19'+j)
    
print len(year_list_train)
#print year_list

year_train=pd.DataFrame({'year_list':year_list_train})
year_df_train=year_train['year_list'].astype(int).tolist()

plt.hist(year_df_train)
plt.show()

age_train=[]
for i in year_df_train:
    result=2016-i
    age_train.append(result)

plt.hist(age_train)
plt.show()    


#*****************************TRAIN*********************************
train['DOB']=train['DOB'].fillna('01/01/88')
test['DOB']=train['DOB'].fillna('01/01/88')

print train.isnull().sum()

#******************************Train dataset age filling**********************
dob_train=train['DOB'].astype(str).tolist()
print len(dob_train)
new_list_train=[]
for i in dob_train:
    
    result=re.findall(r'\d{2}/\d{2}/(\d{2})', i)
    new_list_train.append(result)

print len(new_list_train)

year_list_train=[]
for i in new_list_train:
    for j in i:
      year_list_train.append('19'+j)
    
print len(year_list_train)
#print year_list

year_train=pd.DataFrame({'year_list':year_list_train})
year_df_train=year_train['year_list'].astype(int).tolist()

plt.hist(year_df_train)
plt.show()

age_train=[]
for i in year_df_train:
    result=2016-i
    age_train.append(result)
    
print len(age_train) 

#***********************filling the age value in test data********************
dob_test=test['DOB'].astype(str).tolist()
print len(dob_test)
new_list_test=[]
for i in dob_test:
    
    result=re.findall(r'\d{2}/\d{2}/(\d{2})', i)
    new_list_test.append(result)

print len(new_list_test)

year_list_test=[]
for i in new_list_test:
    for j in i:
      year_list_test.append('19'+j)
    
print len(year_list_test)
#print year_list

year_test=pd.DataFrame({'year_list':year_list_test})
year_df_test=year_test['year_list'].astype(int).tolist()

plt.hist(year_df_test)
plt.show()

age_test=[]
for i in year_df_test:
    result=2016-i
    age_test.append(result)
    
print len(age_test) 

##*****************NOW CONCATENATING THE AGE TO THE BOTH THE DATASET***************
train['age']=age_train 
test['age']=age_test 
print train.info() 
print test.info()

train.drop(['DOB','Lead_Creation_Date'], axis=1, inplace=True)
test.drop(['DOB', 'Lead_Creation_Date'], axis=1, inplace=True)



#***********************City Category******************************
#filling the missing value

train['City_Category']=train['City_Category'].fillna('A')
test['City_Category']=test['City_Category'].fillna('A')    
print train.isnull().sum()

#mapping of city category
#A:0
#B:1
#C:2

city_category_mapping={'A':0, 'B':1, 'C':2}
for dataset in train_test_dataset:
    dataset['City_Category']=dataset['City_Category'].map(city_category_mapping)
    
#****************************Employer Category1*******************
#filling the misssing valuses

train['Employer_Category1']=train['Employer_Category1'].fillna('A')
test['Employer_Category1']=test['Employer_Category1'].fillna('A')
print train.isnull().sum()
    
#mapping of employer category
#A:0
#B:1
#C:2

employer_cat1_mapping={'A':0, 'B':1, 'C':2}
for dataset in train_test_dataset:
    dataset['Employer_Category1']=dataset['Employer_Category1'].map(employer_cat1_mapping)
    
#*****************************Emloyer category 2*************************    
#filling the missing values
train['Employer_Category2']=train['Employer_Category2'].fillna(4.0)
test['Employer_Category2']=test['Employer_Category2'].fillna(4.0)    
print train.isnull().sum()

#*********************Monthly Income*******************
#Visualsing the monthly income
facet=sns.FacetGrid(train, hue='Approved', aspect=4)
facet.map(sns.kdeplot, 'Monthly_Income', shade=True)
facet.set(xlim=(0, 4500))
facet.add_legend()
plt.show()

print train['Monthly_Income'].min(), train['Monthly_Income'].max()


#mapping of the monthly income
#0-5000:0
#5000-20000:1
#>20000:2

for dataset in train_test_dataset:
    dataset.loc[dataset['Monthly_Income']<5000, 'Monthly_Income']=0,
    dataset.loc[(dataset['Monthly_Income']>=5000) & (dataset['Monthly_Income']<20000), 'Monthly_Income']=1,
    dataset.loc[dataset['Monthly_Income']>=20000, 'Monthly_Income']=2
    
#***************************Primary_Bank_Type************************
#filling the missing value
train['Primary_Bank_Type']=train['Primary_Bank_Type'].fillna('P')
test['Primary_Bank_Type']=test['Primary_Bank_Type'].fillna('P')   
print train.isnull().sum()
 
#mapping the primary_bank_type
#P:0
#G:1
primary_bank_type_mapping={'P':0, 'G':1}
for dataset in train_test_dataset:
    dataset['Primary_Bank_Type']=dataset['Primary_Bank_Type'].map(primary_bank_type_mapping)

#************************Conatacted***********************
#mapping of cantact 
#Y:0
#N:1
contact_mapping={'Y':0, 'N':1}
for dataset in train_test_dataset:
    dataset['Contacted']=dataset['Contacted'].map(contact_mapping)
    

#***************************Source Category**************************
#mapping of source category
source_category_mapping={'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
for dataset in train_test_dataset:
    dataset['Source_Category']=dataset['Source_Category'].map(source_category_mapping)

#***********************Existing EMI***********************
#Visualsing the monthly income
facet=sns.FacetGrid(train, hue='Approved', aspect=4)
facet.map(sns.kdeplot, 'Existing_EMI', shade=True)
facet.set(xlim=(1000, 10000))
facet.add_legend()
plt.show()

#filling the missing value
train['Existing_EMI'].fillna(train.groupby('Monthly_Income')['Existing_EMI'].transform('median'), inplace=True)
test['Existing_EMI'].fillna(test.groupby('Monthly_Income')['Existing_EMI'].transform('median'), inplace=True)

#mapping the Existing_EMI
for dataset in train_test_dataset:
    dataset.loc[dataset['Existing_EMI']<75, 'Existing_EMI']=0,
    dataset.loc[(dataset['Existing_EMI']>=75) & (dataset['Existing_EMI']<550), 'Existing_EMI']=1,
    dataset.loc[(dataset['Existing_EMI']>=550) & (dataset['Existing_EMI']<1000), 'Existing_EMI']=2,
    dataset.loc[dataset['Existing_EMI']>=1000, 'Existing_EMI']=3

print train['Existing_EMI']
print train.isnull().sum()


#**********************************Loan Ammount****************************
#filling the missing values on the basis of Monthly_Income
train['Loan_Amount'].fillna(train.groupby('Monthly_Income')['Loan_Amount'].transform('median'), inplace=True)
test['Loan_Amount'].fillna(test.groupby('Monthly_Income')['Loan_Amount'].transform('median'), inplace=True)

#Visualising the loan ammount
facet=sns.FacetGrid(train, hue='Approved', aspect=4)
facet.map(sns.kdeplot, 'Loan_Amount', shade=True)
facet.set(xlim=(70000, 200000))
facet.add_legend()
plt.show()


#mapping of loan_Ammount
for dataset in train_test_dataset:
    dataset.loc[dataset['Loan_Amount']<27000, 'Loan_Amount']=0,
    dataset.loc[(dataset['Loan_Amount']>=27000) & (dataset['Loan_Amount']<32500), 'Loan_Amount']=1,
    dataset.loc[(dataset['Loan_Amount']>=32500) & (dataset['Loan_Amount']<70000), 'Loan_Amount']=2,
    dataset.loc[(dataset['Loan_Amount']>=70000) & (dataset['Loan_Amount']<120000), 'Loan_Amount']=3,
    dataset.loc[dataset['Loan_Amount']>=120000, 'Loan_Amount']=4

print train.isnull().sum()

#*****************Loan Period*******************************
#filling the missing value
train['Loan_Period'].fillna(train.groupby('Monthly_Income')['Loan_Period'].transform('median'), inplace=True)
test['Loan_Period'].fillna(test.groupby('Monthly_Income')['Loan_Period'].transform('median'), inplace=True)

#Visualising the Loan_Period
facet=sns.FacetGrid(train, hue='Approved', aspect=4)
facet.map(sns.kdeplot, 'Loan_Period', shade=True)
facet.set(xlim=(1, train['Loan_Period'].max()))
facet.add_legend()
plt.show()

#***************Interest_Rate****************************
train['Interest_Rate'].fillna(train.groupby('Loan_Amount')['Interest_Rate'].transform('median'), inplace=True)
test['Interest_Rate'].fillna(test.groupby('Loan_Amount')['Interest_Rate'].transform('median'), inplace=True)

#Visualising the Interest_Rate
facet=sns.FacetGrid(train, hue='Approved', aspect=4)
facet.map(sns.kdeplot, 'Interest_Rate', shade=True)
facet.set(xlim=(18.7, 37))
facet.add_legend()
plt.show()


train['Interest_Rate'].max()

#mapping of Interest rate
for dataset in train_test_dataset:
    dataset.loc[dataset['Interest_Rate']<17.9, 'Interest_Rate']=0,
    dataset.loc[(dataset['Interest_Rate']>=17.9) & (dataset['Interest_Rate']<18.7), 'Interest_Rate']=1,
    dataset.loc[dataset['Interest_Rate']>=18.7, 'Interest_Rate']=2
    

#**************************EMI****************************
#filling the missing values
train['EMI'].fillna(train.groupby('Loan_Amount')['EMI'].transform('median'), inplace=True)
test['EMI'].fillna(test.groupby('Loan_Amount')['EMI'].transform('median'), inplace=True)

#Visuslising the EMI

facet=sns.FacetGrid(train, hue='Approved', aspect=4)
facet.map(sns.kdeplot, 'EMI', shade=True)
#facet.set(xlim=(train['EMI'].min(), train['EMI'].max()))
facet.set(xlim=(1000, 4000))
facet.add_legend()
plt.show()


#mapping of EMI
for dataset in train_test_dataset:
    dataset.loc[dataset['EMI']<500, 'EMI']=0,
    dataset.loc[(dataset['EMI']>=500) & (dataset['EMI']<1000), 'EMI']=1,
    dataset.loc[(dataset['EMI']>=1000) & (dataset['EMI']<4000), 'EMI']=2,
    dataset.loc[dataset['EMI']>=4000, 'EMI']=3


print test.isnull().sum()


#**************************Age**********************
#Visualising the Age
facet=sns.FacetGrid(train, hue='Approved', aspect=4)
facet.map(sns.kdeplot, 'age', shade=True)
facet.set(xlim=(55, 70))
facet.add_legend()
plt.show()


#mapping of age
for dataset in train_test_dataset:
    dataset.loc[dataset['age']<29, 'age']=0,
    dataset.loc[(dataset['age']>=29) & (dataset['age']<56), 'age']=1,
    dataset.loc[dataset['age']>=56, 'age']=2

print train.isnull().sum()
print test.isnull().sum()

#####################################################################
drop_features=['City_Code', 'Employer_Code', 'Customer_Existing_Primary_Bank_Code', 'Approved', 'ID', 'Source']
train_data=train.drop(drop_features, axis=1)
test_data=test.drop(['City_Code', 'Employer_Code', 'Customer_Existing_Primary_Bank_Code', 'ID', 'Source'], axis=1)
target=train['Approved']


#Importing classifier Modules
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#*******************Applying cross validation**************************
K_fold=KFold(n_splits=10, shuffle=True, random_state=0)

#***************Knn*****************
clf=KNeighborsClassifier(n_neighbors=13)
scoring='roc_auc'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring)

print round(np.mean(score)*100)

#*****************************Decision Tree*************************

clf=DecisionTreeClassifier(max_depth=3, criterion='gini')
scoring='roc_auc'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print score

#decision tree score
print round(np.mean(score)*100)

#*************************Random Forest************************
clf=RandomForestClassifier(n_estimators=13)
scoring='roc_auc'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print score

#random forest score
print round(np.mean(score)*100)


#*****************************Naive Bayes*******************
clf=GaussianNB()
scoring='roc_auc'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print  score

#GuassianNB score
print round(np.mean(score)*100)

#**************************SVM*********************
clf=SVC()
scoring='roc_auc'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print  score

#SVM score
print round(np.mean(score)*100)

#***********************Testing****************************
clf_1=GaussianNB()
clf_1.fit(train_data, target)

predict=clf_1.predict(test_data)

#***************Submission**************************


ID=test['ID']
submission_df=pd.DataFrame({'Approved':predict, 'ID':ID})
submission_df.to_csv('GaussianNB2.csv', index=False)















































