import pandas as pd
import datetime as dt
import numpy as np

def age(b):
    t = dt.date.today()
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))

# Load csvs into dataframes
enrollment_df = pd.read_csv("data/enrollment.csv")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
events_df = pd.read_csv("data/events.csv")

# Drop unnecessary columns from training and events data
train_df.drop(columns=['Index Id', 'Hospital Id', 'Registration Date', 'Procedure Date'], inplace=True)
test_df.drop(columns=['Index Id', 'Hospital Id', 'Registration Date', 'Procedure Date'], inplace=True)
events_df.drop(columns=['Hospital Id', 'Event_Date', 'Event_Time', 'Event_Name', 'Event_Desc'], inplace=True)

# Compute difference (in days) between Registration Date and Procedure Date
enrollment_df['Registration Date'] = pd.to_datetime(enrollment_df['Registration Date'])
enrollment_df['Procedure Date'] = pd.to_datetime(enrollment_df['Procedure Date'])
enrollment_df['Date Diff'] = enrollment_df.apply(lambda row: (row['Procedure Date'] - row['Registration Date']).days, axis=1)
# Convert boolean strings to integer values
enrollment_df['Email'] = enrollment_df['Email'].apply(lambda x: x*1)
enrollment_df['SMS'] = enrollment_df['SMS'].apply(lambda x: x*1)
enrollment_df['Gender'] = enrollment_df['Gender'].apply(lambda x: x == 'Male').astype(int)
enrollment_df['Gender'].fillna(np.rint(enrollment_df['Gender'].mean()), inplace=True)

# Compute age from date of birth
enrollment_df['Date of Birth'] = pd.to_datetime(enrollment_df['Date of Birth'])
enrollment_df['Age'] = enrollment_df['Date of Birth'].apply(lambda x: age(x))
enrollment_df['Age'].fillna(np.rint(enrollment_df['Age'].mean()), inplace=True)

# Add columns for the day of week and month of procedure date
enrollment_df['Procedure Weekday'] = enrollment_df['Procedure Date'].apply(lambda x: x.weekday())
enrollment_df['Procedure Month'] = enrollment_df['Procedure Date'].apply(lambda x: x.month)
# Drop unnecessary columns from enrollment data
enrollment_df.drop(columns=['Hospital Id', 'Registration Date', 'Procedure Date', 'Date of Birth'], inplace=True)
# Drop rows with NaN
enrollment_df.dropna(how='any', inplace=True)

# Aggregate events data
grouped_module = events_df.groupby(['Patient Id', 'Module Id'])
grouped_message = events_df.groupby(['Patient Id', 'Message Id'])
# Enumerate unique Patient Ids, Module Ids, and Message Ids
patients = enrollment_df['Patient Id'].unique()
modules = events_df['Module Id'].unique().astype(str)[1:]
messages = events_df['Message Id'].unique().astype(str)[1:]
# Construct additional dataframes, populate with module and message counts
module_df = pd.DataFrame(0, index=patients, columns=modules)
module_df.index.name = 'Patient Id'
message_df = pd.DataFrame(0, index=patients, columns=messages)
message_df.index.name = 'Patient Id'
for name, group in grouped_module:
    patient = name[0]
    module = name[1]
    if patient in patients:
        module_df.at[patient, str(module)] = group.size
for name, group in grouped_message:
    patient = name[0]
    message = name[1]
    if patient in patients:
        message_df.at[patient, str(message)] = group.size

# Combine training, events, and enrollment data
enrollment_df = enrollment_df.merge(module_df, on='Patient Id')
enrollment_df = enrollment_df.merge(message_df, on='Patient Id')
result_train = enrollment_df.merge(train_df.set_index('Patient Id'), on='Patient Id',)
result_test = enrollment_df.merge(test_df.set_index('Patient Id'), on='Patient Id',)

result_train.to_csv('data/data_train.csv')
result_test.to_csv('data/data_test.csv')
