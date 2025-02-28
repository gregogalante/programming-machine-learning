from Classification import Classification

# Setup paths
x_file_path = 'x.csv'
y_file_path = 'y.csv'

# Setup instance
model = Classification(x_file_path, y_file_path)

# Calculate w using train
w = model.train()

# Plot the predictions and real values
model.plot(w)
