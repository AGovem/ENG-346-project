import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os

# ---------- Data Collection ----------
download_command = "curl -L -o smart-crop-recommendation-dataset.zip https://www.kaggle.com/api/v1/datasets/download/miadul/smart-crop-recommendation-dataset"
zip_file_name = "smart-crop-recommendation-dataset.zip"
try:
	if os.path.exists(zip_file_name):
		pass
	else:
                os.system(download_command)
except Exception as e:
	print(f"ERROR: {e}")

try:
	if os.path.exists("crop-yield.csv"):
		df = pd.read_csv("crop-yield.csv")
	else:
		os.system(f"unzip {zip_file_name} && mv data/crop-yield.csv . && rm -r data")
		df = pd.read_csv("crop-yield.csv")
except Exception as e:
	print(f"ERROR: {e}")

# ---------- DATA EXTRACTION ----------

relevant_columns = [
    'N', 'P', 'K', 'Soil_pH', 'Soil_Moisture',
    'Organic_Carbon', 'Temperature', 'Humidity',
    'Rainfall', 'Sunlight_Hours', 'Wind_Speed',
    'Crop_Type', 'Crop_Yield_ton_per_hectare'
]

df_extracted = df[relevant_columns]
with open("extracted_data.txt","w") as extracted_file:
        extracted_file.write(df_extracted.to_string(index=False))
        extracted_file.close()
#print("#"*30,"\n",df_extracted,"\n","#"*30)

# ---------- DATA CLEANING ----------

missing_values = df_extracted.isnull().sum().sum()
print(f"Total Missing Data: {missing_values}")

if missing_values > 0:
	df_extracted = df_extracted.dropna()

duplicates = df_extracted.duplicated().sum()
print(f"Total Duplicated Data : {duplicates}")

if duplicates > 0:
    df_extracted = df_extracted.drop_duplicates()

print(f"Data Set Size After Cleaning: {len(df_extracted)}\n")
with open("cleaned_data.txt","w") as cleaned_file:
        cleaned_file.write(df_extracted.to_string(index=False))
        cleaned_file.close()

# ---------- NORMALIZATION & VISUALIZATION ---------

numeric_df = df_extracted.select_dtypes(include=[np.number])

plt.figure(figsize=(8, 6))
sns.histplot(df_extracted['Crop_Yield_ton_per_hectare'], kde=True)
plt.title('Crop Yield Distribution')
plt.savefig('Rapor04_Graph_Yield_Dist.png')
print("Graphig Generated: Rapor04_Graph_Yield_Dist.png")

scaler = MinMaxScaler()
numeric_cols = numeric_df.columns.drop('Crop_Yield_ton_per_hectare') 

df_normalized = df_extracted.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df_extracted[numeric_cols])
with open("normalized_data.txt","w") as normalized_file:
        normalized_file.write(df_normalized.to_string(index=False))
        normalized_file.close()

# ---------- FEATURE EXTRACTION, TRAINING, VALIDATION ----------

unique_crops = df_normalized['Crop_Type'].unique()
results = []

print(f"Training models for {len(unique_crops)} different crop type")

for crop in unique_crops:
    crop_data = df_normalized[df_normalized['Crop_Type'] == crop]

    X = crop_data[numeric_cols] # Features
    y = crop_data['Crop_Yield_ton_per_hectare'] # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
                    'Crop': crop,
                    'R2_Score': r2,
                    'RMSE': rmse
                    })
    
    mean_yield = y.mean()
    print(f"  -> (Average Crop Yield For {crop}: {mean_yield:.2f})")
print(f"Model Parameters for {crop}: {model.coef_}")
with open("model_parameters_data.txt","wb") as model_parameters_file:
        model_parameters_file.write(model.coef_.tostring())
        model_parameters_file.close()
results_df = pd.DataFrame(results)
print(results_df)

print("\nAverage R2 Score:", results_df['R2_Score'].mean())

# User Data Input Prediction
ans = input("Do you want to calculate your own conditions crop yield (y/n): ")
if ans == "y":
	pass
else:
	quit()

numeric_cols_to_scale = numeric_df.columns.drop('Crop_Yield_ton_per_hectare')
crops_list = df_extracted['Crop_Type'].unique()
print("please select crop you want to predict yield:")

for i, crop_name in enumerate(crops_list):
    print(f"{i + 1}. {crop_name}")

selected_crop = None
while True:
    try:
        secim = int(input("\nSelection (enter number): "))
        if 1 <= secim <= len(crops_list):
            selected_crop = crops_list[secim - 1]
            break
        else:
            print("please enter number from list")
    except ValueError:
        print("please enter number")

print(f"\n Selected Crop: {selected_crop}")

feature_cols = [col for col in numeric_cols_to_scale] 

target_crop_df = df_extracted[df_extracted['Crop_Type'] == selected_crop]
X_pred = target_crop_df[feature_cols]
y_pred_target = target_crop_df['Crop_Yield_ton_per_hectare']

final_model = LinearRegression()
final_model.fit(X_pred, y_pred_target)

user_inputs = {}

labels = {
        'N': 'Azot (N) Miktarı',
        'P': 'Fosfor (P) Miktarı',
        'K': 'Potasyum (K) Miktarı',
        'Soil_pH': 'Toprak pH (0-14)',
        'Soil_Moisture': 'Toprak Nemi (%)',
        'Organic_Carbon': 'Organik Karbon',
        'Temperature': 'Sıcaklık (°C)',
        'Humidity': 'Nem (%)',
        'Rainfall': 'Yağış (mm)',
        'Sunlight_Hours': 'Güneşlenme Süresi (Saat)',
        'Wind_Speed': 'Rüzgar Hızı',
        'Altitude': 'Rakım (m)',
        'Fertilizer_Used': 'Ekstra Gübre (kg/ha)',
        'Pesticide_Used': 'İlaç Kullanımı (kg/ha)'
    }

for col in feature_cols:
    label = labels.get(col, col) 
    while True:
        try:
            val = input(f"{label} giriniz: ")
            user_inputs[col] = float(val)
            break
        except ValueError:
            print("please enter a number")

input_df = pd.DataFrame([user_inputs])

print("\nCalculating...")
prediction = final_model.predict(input_df)[0]

print("-" * 50)
print(f"Predicted {selected_crop} yield for your conditions:")
print(f"{prediction:.2f} Ton/Hektar")
print("-" * 50)
