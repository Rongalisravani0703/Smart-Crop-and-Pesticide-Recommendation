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


from sklearn.model_selection import train_test_split

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

crop_model.fit(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Crop Model
crop_model = RandomForestClassifier(random_state=42)
crop_model.fit(X_train, y_train)

# Predict on test data
y_pred = crop_model.predict(X_test)

# Evaluate
print("🌾 Crop Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

X_pest = df_pest[['Pest Name', 'Amount_scaled']]

 Features and target for pesticide dataset
X_pest = df_pest[['Pest Name', 'Amount_scaled']]
y_pest = df_pest['Recommended Pesticide']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_pest, y_pest, test_size=0.2, random_state=42
)

# Train Pesticide Model
pest_model = RandomForestClassifier(random_state=42)
pest_model.fit(Xp_train, yp_train)

# Predict
yp_pred = pest_model.predict(Xp_test)

# Evaluate
print("🧪 Pesticide Model Accuracy:", accuracy_score(yp_test, yp_pred))


# Example input [N, P, K, temperature, humidity, ph, rainfall]
import pandas as pd

sample = pd.DataFrame([[90, 42, 43, 20, 80, 6.5, 200]],
                      columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

# Scale with the same scaler
sample_scaled = scaler.transform(sample)

# Predict
predicted_crop = crop_model.predict(sample_scaled)

# Decode back to crop name
predicted_crop_name = encoder.inverse_transform(predicted_crop)
print("✅ Recommended Crop:", predicted_crop_name[0])


# Encode pest name
pest_encoder = LabelEncoder()
df_pest['Pest Name Encoded'] = pest_encoder.fit_transform(df_pest['Pest Name'])

# Encode pesticide
pesticide_encoder = LabelEncoder()
df_pest['Pesticide Encoded'] = pesticide_encoder.fit_transform(df_pest['Recommended Pesticide'])


X_pest = df_pest[['Pest Name Encoded', 'Amount_scaled']]
y_pest = df_pest['Pesticide Encoded']


sample_pest = pd.DataFrame([[3, 1.5]],
                           columns=['Pest Name Encoded', 'Amount_scaled'])

predicted_pesticide = pest_model.predict(sample_pest)

# Decode back to original pesticide name
predicted_pesticide_name = pesticide_encoder.inverse_transform(predicted_pesticide)
print("Recommended Pesticide:", predicted_pesticide_name[0])


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


def recommend_system(N, P, K, temperature, humidity, ph, rainfall, pest_id, amount_scaled):
    # Crop prediction
    crop_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N','P','K','temperature','humidity','ph','rainfall'])
    crop_scaled = scaler.transform(crop_input)
    crop_pred = crop_model.predict(crop_scaled)
    crop_name = encoder.inverse_transform(crop_pred)[0]

    # Pesticide prediction
    pest_input = pd.DataFrame([[pest_id, amount_scaled]],
                              columns=['Pest Name Encoded','Amount_scaled'])
    pesticide_pred = pest_model.predict(pest_input)
    pesticide_name = pesticide_encoder.inverse_transform(pesticide_pred)[0]

    return crop_name, pesticide_name


 Example: Pest name already encoded, amount already scaled
sample_pest = pd.DataFrame([[3, 1.5]], columns=['Pest Name Encoded','Amount_scaled'])
pest_pred = pest_model.predict(sample_pest)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Crop model evaluation
y_crop_pred = crop_model.predict(X_test)
print("Crop Model Accuracy:", accuracy_score(y_test, y_crop_pred))
print("Crop Model Report:\n", classification_report(y_test, y_crop_pred))

# Pesticide model evaluation
yp_pest_pred = pest_model.predict(Xp_test)
print("Pest Model Accuracy:", accuracy_score(yp_test, yp_pest_pred))
print("Pest Model Report:\n", classification_report(yp_test, yp_pest_pred))

import joblib
joblib.dump(crop_model, "crop_model.pkl")
joblib.dump(pest_model, "pest_model.pkl")
joblib.dump(scaler, "scaler.pkl")   # for scaling crop features

crop_model = joblib.load("crop_model.pkl")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict crops on test data
y_crop_pred = crop_model.predict(X_test)

# Accuracy
print("🌱 Crop Model Accuracy:", accuracy_score(y_test, y_crop_pred))

# Classification report
print("\nCrop Classification Report:\n", classification_report(y_test, y_crop_pred))

# Confusion matrix
cm_crop = confusion_matrix(y_test, y_crop_pred)

plt.figure(figsize=(10,6))
sns.heatmap(cm_crop, annot=False, cmap="Blues")
plt.title("Confusion Matrix - Crop Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Predict pesticides on test data
yp_pest_pred = pest_model.predict(Xp_test)

# Accuracy
print("🧪 Pesticide Model Accuracy:", accuracy_score(yp_test, yp_pest_pred))

# Classification report
print("\nPesticide Classification Report:\n", classification_report(yp_test, yp_pest_pred))

# Confusion matrix
cm_pest = confusion_matrix(yp_test, yp_pest_pred)

plt.figure(figsize=(10,6))
sns.heatmap(cm_pest, annot=False, cmap="Greens")
plt.title("Confusion Matrix - Pesticide Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
