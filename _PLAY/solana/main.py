import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This script will use open price of solana, tesla and eur/usd to predict the close price of solana.

# Prepare dataset
#

# Load data from CSV files
solana_price_history_path = 'solana_price_history.csv'
solana_price_history = []
with open(solana_price_history_path) as solana_price_history_file:
  solana_price_history_reader = csv.reader(solana_price_history_file)
  next(solana_price_history_reader)
  for row in solana_price_history_reader:
    solana_price_history.append({
      'date': row[0],
      'open': float(row[1].replace('$', '')),
      'close': float(row[4].replace('$', '')),
    })
tesla_price_history_path = 'tesla_price_history.csv'
tesla_price_history = []
with open(tesla_price_history_path) as tesla_price_history_file:
  tesla_price_history_reader = csv.reader(tesla_price_history_file)
  next(tesla_price_history_reader)
  for row in tesla_price_history_reader:
    tesla_price_history.append({
      'date': row[0],
      'open': float(row[3].replace('$', '')),
    })
eur_usd_price_history_path = 'eur_usd_price_history.csv'
eur_usd_price_history = []
with open(eur_usd_price_history_path) as eur_usd_price_history_file:
  eur_usd_price_history_reader = csv.reader(eur_usd_price_history_file)
  next(eur_usd_price_history_reader)
  for row in eur_usd_price_history_reader:
    eur_usd_price_history.append({
      'date': row[0],
      'value': float(row[1]),
    })

# Normalize data
# convert date string ('MM/DD/YYYY') to datetime object for solana price history 
solana_price_history_normalized = []
for row in solana_price_history:
  date = datetime.datetime.strptime(row['date'], '%m/%d/%Y')
  solana_price_history_normalized.append({
    'date': date,
    'open': row['open'],
    'close': row['close'],
  })
# convert date string ('MM/DD/YYYY') to datetime object for tesla price history
tesla_price_history_normalized = []
for row in tesla_price_history:
  date = datetime.datetime.strptime(row['date'], '%m/%d/%Y')
  tesla_price_history_normalized.append({
    'date': date,
    'open': row['open'],
  })
# convert date string ('M/D/YYYY Monday') to datetime object for eur/usd price history
eur_usd_price_history_normalized = []
for row in eur_usd_price_history:
  date = datetime.datetime.strptime(row['date'], '%m/%d/%Y %A')
  eur_usd_price_history_normalized.append({
    'date': date,
    'value': row['value'],
  })

# Filter date by removing data points that are not in the intersection of all three datasets
solana_price_dates = [row['date'] for row in solana_price_history_normalized]
tesla_price_dates = [row['date'] for row in tesla_price_history_normalized]
eur_usd_price_dates = [row['date'] for row in eur_usd_price_history_normalized]
common_dates = list(set(solana_price_dates) & set(tesla_price_dates) & set(eur_usd_price_dates))
solana_price_history_normalized_filtered = [row for row in solana_price_history_normalized if row['date'] in common_dates]
tesla_price_history_normalized_filtered = [row for row in tesla_price_history_normalized if row['date'] in common_dates]
eur_usd_price_history_normalized_filtered = [row for row in eur_usd_price_history_normalized if row['date'] in common_dates]

# Sort data by date
solana_price_history_normalized_filtered_sorted = sorted(solana_price_history_normalized_filtered, key=lambda x: x['date'])
tesla_price_history_normalized_filtered_sorted = sorted(tesla_price_history_normalized_filtered, key=lambda x: x['date'])
eur_usd_price_history_normalized_filtered_sorted = sorted(eur_usd_price_history_normalized_filtered, key=lambda x: x['date'])

# Prepare final dataset
dataset = {}
dataset['solana_open'] = [row['open'] for row in solana_price_history_normalized_filtered_sorted]
dataset['tesla_open'] = [row['open'] for row in tesla_price_history_normalized_filtered_sorted]
dataset['eur_usd'] = [row['value'] for row in eur_usd_price_history_normalized_filtered_sorted]
dataset['solana_close'] = [row['close'] for row in solana_price_history_normalized_filtered_sorted]

# Init training and testing dataset
#

x_sol_open = np.array(dataset['solana_open']).reshape(-1, 1)
x_tesla_open = np.array(dataset['tesla_open']).reshape(-1, 1)
x_eur_usd = np.array(dataset['eur_usd']).reshape(-1, 1)
y_sol_close = np.array(dataset['solana_close']).reshape(-1, 1)
print('x_sol_open:', x_sol_open.shape)
print('x_tesla_open:', x_tesla_open.shape)
print('x_eur_usd:', x_eur_usd.shape)
print('y_sol_close:', y_sol_close.shape)

X = np.column_stack((np.ones(x_sol_open.size), x_sol_open, x_tesla_open, x_eur_usd)) # NOTE: The first item np.ones(x_sol_open.size) is a rapresentation of the bias!
print('X:', X.shape)

Y = y_sol_close.reshape(-1, 1)
print('Y:', Y.shape)

w = np.zeros((X.shape[1], 1))
print('w:', w.shape)

def predict(X, w):
  return np.matmul(X, w)

def loss(X, Y, w):
  return np.average((predict(X, w) - Y) ** 2)

def gradient(X, Y, w):
  return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

def train(X, Y, w, learning_rate, epochs):
  for i in range(epochs):
    w -= learning_rate * gradient(X, Y, w)
    if i % 100 == 0:
      print('Epoch:', i, 'Loss:', loss(X, Y, w))
  return w

w = train(X, Y, w, 0.00000001, 1000000)
print('Final loss:', loss(X, Y, w))

# Plot the result
#

plt.figure(figsize=(12, 6))
plt.plot(dataset['solana_close'], label='Actual')
plt.plot(predict(X, w), label='Predicted')
plt.legend()
plt.show()
