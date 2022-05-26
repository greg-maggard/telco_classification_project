import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def significance_test(p):
    '''
    Assumes an alpha of .05. Takes in a value, p, and returns a string stating whether or not that p value indicates sufficient evidence
    to reject a null hypothesis.
    '''
    α = 0.05
    if p < α:
        print("Sufficient evidence -> Reject the null hypothesis.")
    else:
        print("Insufficient evidence -> Fail to reject the null hypothesis.")

def optimal_rf_finder(X_train, y_train, X_validate, y_validate, max_depth = 20):
    '''
    Takes in X_train, y_train, X_validate, y_validate, and a default max_depth of 20. Creates many random forest models with 
    max_depth and n_samples that change in an inverse pattern to one another (as max_depth goes up, n_samples goes down). Returns dataframe.
    '''
    # Writing a loop to visualize best model's performance in relation to different hyperparameters:
    metrics = []

    for i in range(2, max_depth):
        # Make the model (note that as n_samples goes up, max_depth will go down)
        depth = max_depth - i
        n_samples = i
        forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=123)

        # Fit the model on the training set to determine accuracy on known data
        forest = forest.fit(X_train, y_train)

        # Evaluate model's performance on training set:
        in_sample_accuracy = forest.score(X_train, y_train)
        
        #Evaluate how the model performs on the validate set to compare with the training set (helps check for overfitting):
        out_of_sample_accuracy = forest.score(X_validate, y_validate)

        #creates an entry into a dictionary containing metrics for each hyperparameter combination:
        output = {
            "min_samples_per_leaf": n_samples,
            "max_depth": depth,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
        }
        
        metrics.append(output)

    # converting output dictionary to dataframe:    
    df = pd.DataFrame(metrics)
    #Adding a 'difference' column that contains values for the difference between the model's accuracy on the train and validate sets:
    df["difference"] = df.train_accuracy - df.validate_accuracy
    return df

def rf_performance_grapher(df):
    '''
    Takes a dataframe of random forest models and returns a graph depicting their performance on in-sample data, out-of-sample data,
    and the difference between those two.
    '''
    #Creating a graph to chart the metrics for each of the above hyperparameter combinations:
    df.set_index('max_depth')[['train_accuracy', 'validate_accuracy','difference']].plot(figsize = (16,9))
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,21,1))
    plt.title('Model Performance as Related to Hyperparameter Selection')
    plt.grid()
        