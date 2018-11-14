import pandas as pd

enrollment_df = pd.read_csv("data/enrollment.csv")
events_df = pd.read_csv("data/events.csv")
reference_df = pd.read_csv("data/reference.csv")


id = [14723, 14825]
print(events_df.loc[(events_df['Patient Id'] == 14723] && events_df['Module Id'] == 911.0)])
