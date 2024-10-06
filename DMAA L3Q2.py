import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('demat_account_dataset.csv')

# Convert the 'Month' column to numeric values
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y').map(lambda x: x.toordinal())

# Split the data into features (X) and target variable (y)
X = data[['Month']]
y = data['DEMAT accounts']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the DEMAT account count for January 2025
jan_2025 = pd.to_datetime('Jan-25', format='%b-%y').toordinal()
prediction = model.predict([[jan_2025]])

print("Predicted DEMAT account count for January 2025:", int(prediction[0]))