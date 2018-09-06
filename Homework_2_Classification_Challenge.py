# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# Importing the dataset - only using
dataset = pd.read_csv('application_train.csv')
X = dataset.iloc[:, 2:11].values
y = dataset.iloc[:, 1].values

#Describing the Dataset

print('Training data shape: ', dataset.shape)
dataset.head()

dataset.describe()
dataset.columns

#Training data shape:  (307511, 122)
#For this exercise, using Predictor Columns 'NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR',
# 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL','AMT_CREDIT', 
#'AMT_ANNUITY','AMT_GOODS_PRICE'

#Exploratory Data Analysis...following along from kaggle 'gentle intro'
#Imbalance noted, more people paid their loans back than those who did not
dataset['TARGET'].value_counts()
dataset['TARGET'].astype(int).plot.hist();

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
# Missing values statistics
missing_values = missing_values_table(dataset.iloc[:, 2:11])
missing_values.head(20)

#OUTPUT: Your selected dataframe has 9 columns.
#There are 2 columns that have missing values.
 
 #                Missing Values  % of Total Values
#AMT_GOODS_PRICE             278                0.1
#AMT_ANNUITY                  12                0.0
 
#Column DataTypes, Number of each type of column b/c can only do ML on numbers
dataset.iloc[:, 2:11].dtypes.value_counts()

#Datatypes Output 
#object     4
#float64    4
#int64      1

# Number of unique classes in each object column
dataset.iloc[:, 2:11].select_dtypes('object').apply(pd.Series.nunique, axis = 0)

#Encoding Categorical Variables
 
# Create a label encoder object
le = LabelEncoder()
le_count = 0

#New variable training_set b/c in loop at line 95 I'm modifying the training set
training_set = dataset.iloc[:, 2:11]

# Iterate through the columns
for col in training_set:
    if training_set[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(training_set[col].unique())) <= 2:
            # Train on the training data
            le.fit(training_set[col])
            # Transform both training and testing data
            training_set[col] = le.transform(training_set[col])
            
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# 3 columns were label encoded
#If I had NOT done label encoding and just used one-hot, then I would have
#expanded the 4 object columns to 12 columns vs 6 columns 

# one-hot encoding of categorical variables
training_set = pd.get_dummies(training_set)

print('Training Features shape: ', training_set.shape)

#Iterate through columns to describe the data and look for outliers

#for col in training_set: 
#    print(training_set[col].describe())
    
#for col in training_set: 
#    (training_set[col].plot.hist())
#    plt.xlabel(col)
#    plt.show()
    
#NAME_CONTRACT_TYPE = SIGNIFICANTLY more Cash Loans than Revolving Loans
#OWN_CAR = ROUGHLY 2/3 DO NOT OWN A CAR
#OWN_REALTY = ROUGHLY2/3 DO OWN REAL ESTATE
#CHILDRENT = MOST PEOPLE HAVE 0 OR 1, THERE IS AT LEAST ONE OUTLIER WITH 19...
#INCOME_TOTAL = SEVERAL OUTLIERS MAKING $10M - $100M
#AMT_CREDIT = MOST PEOPLE HAVE UNDER $100K OF CREDIT
#AMT_ANNUITY = MOST PEOPLE HAVE UNDER $50K
#AMT_GOODS_PRICE = ROUGHLY $100K
#GENDER = ROUGHLY TWICE AS MANY FEMALES
    
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(training_set)
training_set = imputer.transform(training_set)
 

#KERNEL_SVM - KILLED MY COMPUTER

#RANDOM FOREST CLASSIFICATION

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_set, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#CM RESULTS RANDOM FOREST - SEEMS TO DO THE BEST OUT OF THE 3 THAT I RAN

#69444	1343
#5833	258

# Fitting Decision Tree Classification to the Training set *ENTROPY
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#CM RESULTS DECISION TREE -  ONLY SLIGHTLY WORSE THAN RANDOM FOREST
#65850	4937
#5437	654

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#CM RESULTS NAIVE BAYS - THIS IS TERRIBLE
#581	70206
#20	6071



# Visualising the Training set results
#KEEP GETTING THIS ERROR, DON'T KNOW HOW TO FIX
#ValueError: Number of features of the model must match the input. Model n_features is 11 and input n_features is 2 

"""from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""
