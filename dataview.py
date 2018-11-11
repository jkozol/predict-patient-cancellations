import pandas as pd

enrollment_df = pd.read_csv("data/enrollment.csv")
events_df = pd.read_csv("data/events.csv")
reference_df = pd.read_csv("data/reference.csv")

print(enrollment_df.head())
print(events_df.head())
print(reference_df.head())
