#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
data = pd.read_csv(url, sep='|')

# Basic data analysis
print("Dataset overview:")
print(data.info())
print("\nDescriptive statistics:")
print(data.describe())

# Visualization of total cost per region
plt.figure(figsize=(10, 6))
sns.countplot(x='occupation', data=data)
plt.title('Count of Users by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Scatter plot for Age vs. Zip Code
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='zip_code', data=data)
plt.title('Age vs. Zip Code')
plt.xlabel('Age')
plt.ylabel('Zip Code')
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Example data
x = ['January', 'February', 'March', 'April', 'May']
y = [10, 20, 15, 25, 30]

# Create a plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, marker='o')

# Rotate x-axis tick labels
plt.xticks(rotation=90)

# Show the plot
plt.show()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
data = pd.read_csv(url, sep='|')

# Prepare features and target variable
X = data[['age', 'gender', 'zip_code']]
y = data['occupation']

# Convert categorical variable to dummy variables
X = pd.get_dummies(X, columns=['gender', 'zip_code'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[12]:


import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
data = pd.read_csv(url, sep='|')

# Display initial state of the data
print("Initial data:")
print(data.head())

# Handle missing values (if any)
#data.fillna(data.mean(), inplace=True)
# Handle missing values for numeric columns only
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Encode categorical variables
data = pd.get_dummies(data, columns=['gender', 'occupation'])

# Save the cleaned dataset
data.to_csv('cleaned_users.csv', index=False)

# Display cleaned data
print("\nCleaned data:")
print(data.head())

def calculate_fill_rate(demand, supply):
    """
    Calculate the fill rate.
    
    Parameters:
    demand (int): Total demand.
    supply (int): Total supply.
    
    Returns:
    float: Fill rate as a percentage.
    """
    if demand == 0:
        return 0.0
    return (supply / demand) * 100

# Example usage
demand = 1000
supply = 950
fill_rate = calculate_fill_rate(demand, supply)
print(f"Fill Rate: {fill_rate:.2f}%")

# In[2]:


get_ipython().system('jupyter nbconvert --to script Practical Task.ipyn')

