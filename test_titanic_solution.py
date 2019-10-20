# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:54:33 2019

@author: RamenChetia
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#from sklearn.ensemble import randomforestclassifier

train1 = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train1.describe()
train1.shape

train1.info()

#To get the counts of NA's in each columns
pd.isnull(train1).sum()

train1['Age'].mean()
train1['Age'].median()

#Now we are imputing the age column with median

train1.describe()
train1['Age'] = train1['Age'].fillna(train1['Age'].median())
#This statement is returning all the values along with the impiuted NA's
print(train1['Age'].fillna(train1['Age'].median()))

#Now we are checking the visualization

train1['Died'] = 1 - train1['Survived']
train1.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind = 'bar',figsize = (25,7),
               stacked = True,color = ['g','r'])


def status(feature):
    print ('processing',feature,':ok')
    
#Combining the test and training data

def combine_data():
    train1 = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    target = train1.Survived #train1['Survived'] same 
    train1.drop(['Survived'],1,inplace = True)
    combine = train1.append(test)
    combine.reset_index(inplace = True)
    combine.drop(['index','PassengerId'],inplace = True,axis = 1)
    return combine

combined = combine_data()

combined.info()
combined.shape

combined['Name'].head(10)

#created an empty set to fill the titles


title_dictionary = {'Capt':'Officer',
                    'Col':'Officer',
                    "Major": "Officer",
                    "Jonkheer": "Royalty",
                    "Don": "Royalty",
                    "Sir" : "Royalty",
                    "Dr": "Officer",
                    "Rev": "Officer",
                    "the Countess":"Royalty",
                    "Mme": "Mrs",
                    "Mlle": "Miss",
                    "Ms": "Mrs",
                    "Mr" : "Mr",
                    "Mrs" : "Mrs",
                    "Miss" : "Miss",
                    "Master" : "Master",
                    "Lady" : "Royalty"}

#Just checking how split is working with the index
#naam = 'Braund, Mr. Owen Harris,Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg),Nasser, Mrs. Nicholas (Adele Achem)'    
#print(naam.split(',')[3])  
    
def get_titles():
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    combined['Title'] = combined.Title.map(title_dictionary)
    status('Title')
    return combined

combined = get_titles()

combined.info()
#Checking if we have a title called officer
#filter = combined['Title'] == 'Officer'
#combined.where(filter,inplace = True)    
    
combined.head()    

#Checking the null value w.r.t. Title
combined[combined['Title'].isnull()]  

#Checking the age

print(combined['Age'].iloc[:891].isnull().sum())

print(combined['Age'].iloc[891:].isnull().sum())

#Doing a groupby to find out the median of grouped classes
combined.info()
grouped_items = combined.iloc[:891].groupby(['Sex','Pclass','Title'])

print(list(grouped_items))

grouped_median_train = grouped_items.median()

grouped_median_train = grouped_median_train.reset_index()[['Sex','Pclass','Title','Age']]

#Now we have to impute the age data

def fill_age(row):
    condition = (
    (grouped_median_train['Sex'] == row['Sex']) &
     (grouped_median_train['Title']==row['Title']) &
     (grouped_median_train['Pclass'] == row['Pclass'])
                 )
    return combined['Age'].values[0]

def process_age():
    global combined
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'],axis =1)
    status('age')
    return combined

combined = process_age()

combined['Age']

# Now we are processing the Name column

def process_name():
    global combined
    combined.drop(['Name'],axis = 1,inplace = True)
    #encoding dummies
    title_dummies = pd.get_dummies(combined['Title'],prefix = 'Title')
    combined = pd.concat((combined,title_dummies),axis = 1)
    combined.drop('Title',axis = 1,inplace = True)
    status('name')
    return combined

combined = process_name()

combined.info()

#Processing the fares which has Nan values

def process_fare():
    combined['Fare'].fillna(combined.iloc[:891].Fare.median(),inplace = True)
    status('fare')
    return combined

combined = process_fare()

def process_embark():
    global combined
    combined.Embarked.fillna('S',inplace = True)
    #encoding dummies
    Embarked_dummies = pd.get_dummies(combined['Embarked'],prefix = 'Embarked')
    combined = pd.concat([combined,Embarked_dummies],axis = 1)
    combined.drop('Embarked',axis = 1,inplace = True)
    status('embarked')
    return combined

combined = process_embark()

#Processing cabin

train_cabin,test_cabin = set(),set()

for i in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(i[0])
    except:
        train_cabin.add('U')
        
for i in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(i[0])
    except:
        test_cabin.add('U')
        
        
print(test_cabin)

def process_cabin():
    global combined
    combined.Cabin.fillna('U', inplace = True)
    combined['Cabin'] = combined['Cabin'].map(lambda i:i[0])
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix = 'Cabin')
    combined = pd.concat([combined,cabin_dummies],axis = 1)
    combined.drop('Cabin',axis = 1,inplace =True)
    status('cabin')
    return combined

combined = process_cabin()
combined.info()

###Processing sex

def process_sex():
    global combined
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    status('sex')
    return combined


combined = process_sex()
combined['Sex'].head(10)

##Process Pclass

def process_pclass():
    global combined
    combined_dummies = pd.get_dummies(combined['Pclass'],prefix = 'Pclass')
    combined = pd.concat([combined,combined_dummies],axis = 1)
    combined.drop('Pclass',axis = 1,inplace = True)
    status('pclass')
    return combined

combined = process_pclass()

#Processing tickets

#combined['Ticket'].head(10)
#
#def process_ticket():
#    
#    global combined
#    
#    def cleanTicket(ticket):
#        ticket = ticket.replace('.','')
#        ticket = ticket.replace('/','')
#        ticket = ticket.split()
#        ticket = map(lambda t : t.strip(), ticket)
#        ticket = filter(lambda t : not t.isdigit(), ticket)
#        if len(ticket) > 0:
#            return ticket[0]
#        else: 
#            return 'XXX'
#        
#tickets = set()
#for t in combined['Ticket']:
#    tickets.add(cleanTicket(t))
#        
#    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
#    combined_dummies = pd.get_dummies(combined['Ticket'],prefix = 'Ticket') 
#    combined = pd.concat([combined,combined_dummies],axis = 1)
#    combined.drop('Ticket',axis = 1,inplace = True)
#    status('ticket')
#    return combined
#
#combined = process_ticket()

#Processin family

def process_family():
    global combined
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    combined['Singleton'] = combined['FamilySize'].map(lambda s:1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s:1 if s <= 2 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s:1 if 5 <= s else 0)
    status('family')
    return combined

combined = process_family()

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def compute_score(clf,X,y,scoring = 'accuracy'):
    xval = cross_val_score(clf,X,y,cv = 5,scoring = scoring)
    return np.mean(xval)


    targets = pd.read_csv('train.csv',usecols = ['Survived'])['Survived'].values
    train = combined.iloc[:891]
    test = combined.iloc[891:]
    

clf = RandomForestClassifier(n_estimators = 50,max_features = 'sqrt')
clf = clf.fit(train, targets)

train.info()
test.info()
train.drop('Sex',axis = 1,inplace = True)
test.drop('Age',axis = 1,inplace = True)
#train['Sex'].head(2)
#train.info()

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by = ['importance'],ascending = True,inplace = True)
features.set_index('feature',inplace = True)

features.plot(kind = 'barh',figsize=(25,25))

model = SelectFromModel(clf,prefit = True)
train_reduce = model.transform(train)
train_reduce.shape

test_reduce = model.transform(test)
test_reduce.shape

#MODEL BUILDING

logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg,logreg_cv,rf,gboost]

for model in models:
    print ('Cross-validation of :{0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduce, y=targets, scoring='accuracy')
    print ('CV score {0}='.format(score))
    print ('****')





    