import pandas as pd
import numpy as np
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

#Clean Titanic Data Files:

def clean_titanic_data(df):
    '''
    Takes in a titanic dataframe and returns a cleaned dataframe
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Returns: clean_df - a dataframe with the cleaning operations performed on it
    '''
    #Drop Duplicates
    df.drop_duplicates(inplace = True)
    #Drop Columns
    columns_to_drop = ['embarked', 'class', 'passenger_id', 'deck']
    df = df.drop(columns = columns_to_drop)
    #Encoded Categorical Variables
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df.drop(columns =  ['sex', 'embark_town'])

def impute_age(train, validate, test):
    '''
    Imputes the mean age of train to all three datasets.
    '''
    imputer = SimpleImputer(strategy = 'mean', missing_values = np.nan)
    imputer = imputer.fit(train[['age']])
    train[['age']] = imputer.transform(train[['age']])
    validate[['age']] = imputer.transform(validate[['age']])
    test[['age']] = imputer.transform(test[['age']])
    return train, validate, test

def prep_titanic_data(df):
    df = clean_titanic_data(df)
    train, test = train_test_split(df, train_size = 0.8, stratify = df.survived, random_state = 1234)
    train, validate = train_test_split(train, train_size = .7, stratify = train.survived, random_state = 1234)
    train, validate, test = impute_age(train, validate, test)
    return train, validate, test

#--------------------------------------------------------------------------------------------------------------------------------------------

#Cleaning Iris Data:
def clean_iris_data(df):
    #Dropping unneeded columns
    columns_to_drop = ['species_id', 'measurement_id']
    df = df.drop(columns = columns_to_drop)
    #renaming species_name column
    df = df.rename(columns = {'species_name' : 'species'})
    #creating dummy variables of species
    dummy_df = pd.get_dummies(df[['species']], drop_first = True)
    #concatenating dummy variables onto original dataframe
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def prep_iris_data(df):
    df = clean_iris_data(df)
    train, test = train_test_split(df, train_size = 0.8, stratify = df.species, random_state = 1234)
    train, validate = train_test_split(train, train_size = 0.8, stratify = train.species, random_state = 1234)
    return train, validate, test

    #--------------------------------------------------------------------------------------------------------------------------------------------

#Prepare Telco Data:
def clean_telco_data(df):
    '''
    Cleans the Telco data by converting senior_citizen to string values, casting total_charges to float values, dropping foreign key columns,
    and creating dummy values for categorical columns that can ultimately be used for modeling.
    '''
    #Replacing empty cells with nulls:
    df = df.replace(' ', np.nan)
    #Replacing ints with yes or no:
    df.senior_citizen = df.senior_citizen.replace([0, 1], ['No', 'Yes'])
    #Changing total_charges from 'object' type to 'float64':
    df.total_charges = df.total_charges.astype('float')
    #Dropping rows with nulls:
    df = df.dropna()
    #Dropping unneeded columns:
    columns_to_drop = ['contract_type_id', 'payment_type_id', 'internet_service_type_id']
    df = df.drop(columns = columns_to_drop)
    #creating dummy variables of categorical columns
    dummy_df = pd.get_dummies(df[['gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'contract_type', 'payment_type', 'internet_service_type']], drop_first = True)
    #concatenating dummy variables onto original dataframe
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def prep_telco_data(df):
    '''
    Prepares the cleaned Telco data by splitting it into train, validate, and test sets to be used for exploration. Returns train, validate, test.
    '''
    df = clean_telco_data(df)
    train, test = train_test_split(df, train_size = 0.8, stratify = df.churn, random_state = 1234)
    train, validate = train_test_split(train, train_size = 0.8, stratify = train.churn, random_state = 1234)
    return train, validate, test

def telco_sample_splitter(train, validate, test):
    '''
    Splits train, validate, and test subsets into x and y sets for modeling and validation. Returns X_train, y_train, X_validate, y_validate, X_test, y_test.
    '''
    #Creating Train Set:
    X_train = train.drop(columns = ['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection',
    'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'monthly_charges', 'total_charges', 'churn', 'contract_type', 'payment_type', 'internet_service_type'])
    y_train = train.churn

    #Creating Validate Set:
    X_validate = validate.drop(columns = ['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection',
    'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'monthly_charges', 'total_charges', 'churn', 'contract_type', 'payment_type', 'internet_service_type'])
    y_validate = validate.churn

    #Creating Test Set:
    X_test = test.drop(columns = ['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection',
    'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'monthly_charges', 'total_charges', 'churn', 'contract_type', 'payment_type', 'internet_service_type'])
    y_test = test.churn
    return X_train, y_train, X_validate, y_validate, X_test, y_test

