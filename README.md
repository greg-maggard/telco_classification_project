# Telco Classification Project

## Quick Summary of README.md
- Q: Who is a customer that churns? A: A customer with fiber optic internet, and those who pay with electronic checks.
- Q: How to prevent churn? A: A good initial step would be to improve quality of life for fiber optics customers by focusing on quality of experienece with fiber and customer service. Further, incentivizing a shift away from paying with electronic checks to other automated methods. 

## Contents of README.md:
1. Introduction
2. Goals and Deliverables
3. Initial Hypothesis/Questions
4. Process
6. Key Findings
7. Recommendations
8. Final take-aways and future possibilities
9. Appendices
	1. Data Dictionary
	2. Module Descriptions
	3. Reproducing this project

## Introduction

This project provides exposure to the entire data science pipeline. This includes steps to acquire, prepare, explore, model, and interpret data in order to understand which variables lead to churn at Telco so that better predictive models can be created, and churn can ultimately be reduced. 

## Goals and Deliverables

- Goals
	- Identify at least 3 variables that predict churn
	- A model that beats the baseline model's performance in predicting churn. 
	- A report communicating the steps taken within the data science pipeline to determine drivers of churn and build models.
- Deliverables
	- This README file.
	- A final report Jupyter Notebook containing annotated code and visualizations that convey drivers of churn and recommendations to reduce it. 
	- Python modules automating the acquire and preparation of the data
	- A CSV of data predictions

## Initial Hypotheses:

- I expect customers paying by electronic check to churn at significantly higher rates.
- I expect customers with multiple lines on their account to churn at significantly higher rates. 
- I expect customers with fiber optic internet to churn at significantly higher rates. 

## Further Questions:
- What further analysis can be done to establish clear causal relationships between some of these features and churn?
- Coudl we acquire data on customer satisfaction, in order to understand directly the pain points that lead a customer to churn?

## Process

#### Acquire and Prepare
- Data is acquired from the CodeUp mySQL database using credentials stored in env.py
- see acquire.py and prepare.py for full details on code, and acquire_prepare_notes.ipynb for more detailed notes
- foreign key columns were dropped 
- total_charges was cleaned and cast to float
	- total_charges contained 11 rows that had `' '` as a value.  These all had `tenure = 0`; because there were few rows in a large dataset, they were dropped. These values indicated that the customer likely had not been with Telco for a full billing cycle. 
- prepare.py contains split_telco_data() that will split the data into train, validate and test

#### Data Exploration
- Things to look at:
    - Senior Citizens? - This relationship seems iffy
    - Partnered
    - Dependents? - This relationship doesn't *appear* as strong as whether someone is partnered
    - Multiple lines - Appears multi-line accounts are less likely to churn
    - Add-ons seem like a major predictor (however, I want to be sure I'm not assigning causal relationship where there is only correlation. Could these and churn be driven by some other shared cause e.g. contract type?)
        - online_security: those who don't have it are MUCH more likely to churn, despite it being less common to have it
        - online_backup: those who don't have it are more likely to churn, despite it being less common to have it
        - device_protection: those who do not have it are much more likely to churn, despite it being less common to have it. 
        - tech_support: those who do not have it are much more likely to churn, despite it being less common to have it. 
        - streaming_tv: There may be something here, but I do want to check the degree to which it impacts churn
        - streaming_movies: seems similar to TV, so I need to check this, too. 
    - paperless_billing: there seems to be a relationship.
        - Is there a contract type that typically goes paperless? Is this a driver, or tied to another shared cause?
    - contract_type: this appears to be a major predictor of churn. Monthly churn is incredibly high, while 1 and 2 year are naturally much lower (they have 1/12 or 1/24 chance of being able to churn here - doesn't necessarily imply they're happier.)
    - payment_type: customers who pay by electronic check are much more likely to churn than those with other methods
    - internet_service_type: Those who churn are much more likely to have had fiber optic service. 

#### Hypothesis Testing
- It turns out that while electronic payments and fiber optic internet were good predictors of churn, multiple line accounts were not.

#### Models
- The following are good candidates for models:
	- Random Forest with a max_depth of 10 and a min_sample_leaf of 10
	- Decision Tree with a max_depth of 4
	- K Nearest Neighbors with k = 18
- Overall the best performer is the Random Forest

## Key Findings
- Payment type was a predictor of churn
- Fiber optic internet was a predictor of churn
- Multiple line accounts were NOT predictors of churn
- The Random Forest proved to be the best model, overall. 

## Recommendations

- Advocating for customers to sign up for a payment method other than electronic check.
- Emphasize improving quality of life for customers with fiber optic service. 

## Final Takeaways

 - Payment type and fiber optic internet are good predictors of churn, while having multiple lines is not. 
- Using the random forest model, I was able to achieve an ~82% accuracy in predicting whether a customer would churn, which beats the baseline model by ~ 8.6%.
- I would expect this model to reasonably perform with about 80% accuracy on future data.
    
### Meeting the goals of the project
- The two best predictors analyzed were whether a customer had fiber optic internet, and whether they paid by electronic check.
- The random forest model beats the baseline in terms of predictive accuracy.
- This report communicates these findings
    
### For further research
- Run some more multivariate analysis to understand which features have legitimate causal relationships with churn. Understanding this better may help improve model's performance. 
- Gather data on customer satisfaction, particularly with services like fiber optic internet. This may help understand why having these services seem to indicate a customer will churn.

## Appendices

### Data Dictionary

#### Customer Identification and Demographic Data:
- Customer ID (String)
- Gender (Male/Female)
- Partner status (Bool)
- Dependent status (Bool)
- Senior citizen status (Bool)

#### Customer Relationship information:
- Tenure in months (float)
- Monthly charges (\$USD) (float)
- Total charges (\$USD) (float)
- Paperless Billing (Bool)
- Payment type (categorical)
- Phone Service, with service option columns:
    - Multiple lines : One Line, Multiple Lines, No Phone Service (categorical)
- Internet Service Type: Fiber Optic, DSL, None (categorical)
- Internet Service Option columns (all bool):
    - Online security
    - Online backup
    - Device protection
    - Tech support
    - Streaming TV
    - Streaming movies
- Churn status (bool)

### Custom Module descriptions

- acquire.py
	- get_telco_data(query_db=False) : Retrieves telco data either from the SQL database, or from 'telco.csv' in the current directory. query_db=True will force a database query, and overwrite any .csv file in the current directory
- prepare.py
  - clean_telco_data(df): replaces empty cells with nulls, drops nulls, drops unnecessary columns, creates dummy columns for modeling, changes total_charges type from object to float
	- prep_telco(df) : splits data into train, validate, and test

### Reproducing this project

In order to reproduce this project download `acquire.py`, `prepare.py`, and `final_report.ipnyb`. Then make an `env.py` modeled on the above `env_example.py`.  Then run the `final_report.ipnyb` file in Jupyter Notebook.
