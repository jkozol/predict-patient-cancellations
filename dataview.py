import pandas as pd
import datetime as dt

def age(b):
    t = dt.date.today()
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))

# Load csvs into dataframes
enrollment_df = pd.read_csv("data/enrollment.csv")
train_df = pd.read_csv("data/train.csv")

# Drop unnecessary columns from training data
train_df.drop(columns=['Registration Date', 'Procedure Date'], inplace=True)

#Compute difference (in days) between Registration Date and Procedure Date
enrollment_df['Registration Date'] = pd.to_datetime(enrollment_df['Registration Date'])
enrollment_df['Procedure Date'] = pd.to_datetime(enrollment_df['Procedure Date'])
enrollment_df['Date Diff'] = enrollment_df.apply(lambda row: (row['Procedure Date'] - row['Registration Date']).days, axis=1)
#Convert boolean strings to integer values
enrollment_df['Email'] = enrollment_df['Email'].apply(lambda x: x*1)
enrollment_df['SMS'] = enrollment_df['SMS'].apply(lambda x: x*1)
enrollment_df['Gender'] = enrollment_df['Gender'].apply(lambda x: x == 'Male').astype(int)
#Compute age from date of birth
enrollment_df['Date of Birth'] = pd.to_datetime(enrollment_df['Date of Birth'])
enrollment_df['Age'] = enrollment_df['Date of Birth'].apply(lambda x: age(x))
# Drop unnecessary columns from enrollment data
enrollment_df.drop(columns=['Hospital Id', 'Registration Date', 'Procedure Date', 'Date of Birth'], inplace=True)
# Drop rows with NaN
enrollment_df.dropna(how='any', inplace=True)

# Combine training and enrollment data
result = enrollment_df.merge(train_df.set_index('Patient Id'), on='Patient Id',)

print(result.head())
result.to_csv('data/data.csv')
