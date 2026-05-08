import pandas as pd

# Load the data
calendar = pd.read_csv('../data/raw/calendar.csv')
sales = pd.read_csv('../data/raw/sales_train_evaluation.csv')
prices = pd.read_csv('../data/raw/sell_prices.csv')

print("Calendar Shape:", calendar.shape)
print("Sales Shape:", sales.shape)
print("Prices Shape:", prices.shape)

# View sales
sales.head()