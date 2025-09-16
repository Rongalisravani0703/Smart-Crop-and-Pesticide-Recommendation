import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Upload and Load Datasets ==========
from google.colab import files

print("Upload Crop_recommendation.csv")
uploaded = files.upload()
df = pd.read_csv("Crop_recommendation.csv")

print("Upload full_pesticide_recommendation.csv")
uploaded = files.upload()
df_pest = pd.read_csv("full_pesticide_recommendation.csv")

# ========== Preprocess Crop Data ==========
df = df.dropna()
crop_encoder = LabelEncoder()
df['label_encoded'] = crop_encoder.fit_transform(df['label'])
X_crop = df.drop(["label", "label_encoded"], axis=1)
y_crop = df["label_encoded"]
crop_scaler = StandardScaler()
X_crop_scaled = crop_scaler.fit_transform(X_crop)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_crop_scaled, y_crop, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(random_state=42)
crop_model.fit(Xc_train, yc_train)

# ========== Preprocess Pesticide Data ==========
df_pest = df_pest.dropna()
pest_encoder = LabelEncoder()
pesticide_encoder = LabelEncoder()
df_pest['PestName_encoded'] = pest_encoder.fit_transform(df_pest['Pest Name'])
df_pest['Pesticide_encoded'] = pesticide_encoder.fit_transform(df_pest['Recommended Pesticide'])
df_pest['Amount'] = df_pest['Amount'].astype(str).str.extract(r'([\d\.]+)').astype(float)
df_pest['Amount'] = df_pest['Amount'].fillna(0)
pest_scaler = StandardScaler()
df_pest['Amount_scaled'] = pest_scaler.fit_transform(df_pest[['Amount']])
X_pest = df_pest[['PestName_encoded', 'Amount_scaled']]
y_pest = df_pest['Pesticide_encoded']
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pest, y_pest, test_size=0.2, random_state=42)
pest_model = RandomForestClassifier(random_state=42)
pest_model.fit(Xp_train, yp_train)

# ========== Inference Functions ==========
def recommend_crop_interactive():
    print("\n--- Crop Recommendation ---")
    try:
        N = float(input("Enter value for N (Nitrogen): "))
        P = float(input("Enter value for P (Phosphorus): "))
        K = float(input("Enter value for K (Potassium): "))
        temperature = float(input("Enter temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        ph = float(input("Enter pH: "))
        rainfall = float(input("Enter rainfall (mm): "))
        user_sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=X_crop.columns)
        user_sample_scaled = crop_scaler.transform(user_sample)
        pred = crop_model.predict(user_sample_scaled)
        crop_name = crop_encoder.inverse_transform(pred)[0]
        print("\n✅ Recommended Crop:", crop_name)
    except Exception as e:
        print("Error in crop input:", e)

def recommend_pesticide_interactive():
    print("\n--- Pesticide Recommendation ---")
    # List available pest names
    print("Available pest names:")
    print(", ".join(pest_encoder.classes_))
    try:
        pest_name = input("Enter pest name from above: ")
        if pest_name not in pest_encoder.classes_:
            print("Pest name not found in list!")
            return
        amount = input("Enter amount (just a number, e.g., 1.5): ")
        amount_num = float(pd.Series(str(amount)).str.extract(r'([\d\.]+)').iloc[0][0])
        pest_num = pest_encoder.transform([pest_name])[0]
        amount_scaled = pest_scaler.transform([[amount_num]])[0][0]
        sample = pd.DataFrame([[pest_num, amount_scaled]], columns=['PestName_encoded', 'Amount_scaled'])
        pred = pest_model.predict(sample)
        pesticide_name = pesticide_encoder.inverse_transform(pred)[0]
        print("\n✅ Recommended Pesticide for pest '{}': {}".format(pest_name, pesticide_name))
    except Exception as e:
        print("Error in pesticide input:", e)

# ========== User Interaction ==========
recommend_crop_interactive()
recommend_pesticide_interactive()
