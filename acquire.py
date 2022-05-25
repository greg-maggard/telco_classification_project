import pandas as pd
import os
from env import get_db_url

def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_db_url('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''SELECT species_name, measurements.* FROM species 
                        JOIN measurements ON species.species_id = measurements.species_id;''', get_db_url('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

def get_telco_data():
    filename = "telco_churn.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''SELECT * FROM customers
                            JOIN contract_types ON customers.contract_type_id = contract_types.contract_type_id
                            JOIN payment_types ON customers.payment_type_id = payment_types.payment_type_id
                            JOIN internet_service_types ON customers.internet_service_type_id = internet_service_types.internet_service_type_id;''', get_db_url('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df