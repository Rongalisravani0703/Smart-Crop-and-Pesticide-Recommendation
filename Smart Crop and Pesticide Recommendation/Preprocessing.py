from google.colab import files
uploaded = files.upload() # select crop_recommendation.csv from your computer
# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
file_name = list(uploaded.keys())[0] # Get the name of the uploaded file
df = pd.read_csv(file_name)   # replace with your file path

# Display first few rows
print("First 5 rows:\n", df.head())

# Show dataset info
print("\nDataset Info:\n")
print(df.info())
# Step 2: Handling missing values

# Check how many missing values are there in each column
print("\nMissing Values Before:\n", df.isnull().sum())

# If there are missing rows, we can drop them
df = df.dropna()

# Show again after cleaning
print("\nMissing Values After:\n", df.isnull().sum())
# Step 3: Encode the 'label' column (crop names)

from sklearn.preprocessing import LabelEncoder

# Create encoder object
encoder = LabelEncoder()

# Convert crop names into numeric values
df['label'] = encoder.fit_transform(df['label'])

# Show first few rows after encoding
print("\nAfter Encoding Label:\n", df.head())

# If you want to see mapping of crops to numbers
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("\nLabel Mapping:\n", label_mapping)
# Step 4: Feature Scaling

from sklearn.preprocessing import StandardScaler

# Separate features (X) and target (y)
X = df.drop("label", axis=1)   # input features (N, P, K, temperature, humidity, ph, rainfall)
y = df["label"]                # target (crop)

# Initialize scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

print("\nFirst 5 rows after Scaling:\n", X_scaled[:5])
#Step 5: Train-Test Split

from sklearn.model_selection import train_test_split

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# select full_pesticides_recommendation.csv from your computer
# Step 1: Import required libraries
from google.colab import files
uploaded = files.upload()

# Step 1: Import required libraries
import pandas as pd

# Load the dataset
df_pest = pd.read_csv("full_pesticide_recommendation.csv")

# Display first few rows
print("First 5 rows:\n", df_pest.head())
print("\nDataset Info:\n")
print(df_pest.info())


print("\nMissing Values Before:\n", df_pest.isnull().sum())

df_pest = df_pest.dropna()   # or fill missing if required

print("\nMissing Values After:\n", df_pest.isnull().sum())


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df_pest['Pest Name'] = encoder.fit_transform(df_pest['Pest Name'])
df_pest['Recommended Pesticide'] = encoder.fit_transform(df_pest['Recommended Pesticide'])

print("\nAfter Encoding:\n", df_pest.head())


from sklearn.preprocessing import StandardScaler
import pandas as pd

# Example pest dataset
# df_pest = pd.read_csv("full_pesticide_recommendation.csv")

scaler = StandardScaler()

# Step 1: Remove non-numeric characters (like "L/acre", "kg/ha", etc.)
df_pest['Amount'] = df_pest['Amount'].str.extract(r'([\d\.]+)')  

# Step 2: Convert to float (safe, even if some values were missing)
df_pest['Amount'] = pd.to_numeric(df_pest['Amount'], errors='coerce')

# Step 3: Handle missing values (if any row had no valid number)
df_pest['Amount'] = df_pest['Amount'].fillna(0)

# Step 4: Scale
df_pest['Amount_scaled'] = scaler.fit_transform(df_pest[['Amount']])

print("\nAfter Scaling Amount:\n", df_pest.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


