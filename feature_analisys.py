import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import os

def detect_outliers_mad(series, threshold=3.5):

    mediana = series.median()
    mad = np.median(np.abs(series - mediana))
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z_score = 0.6745 * (series - mediana) / mad
    return np.abs(modified_z_score) > threshold

file_path = 'input/GlobalWeatherRepository.csv'
print("Iniciando processamento de dados...")
df = pd.read_csv(file_path)

cols_vazamento = ['temperature_fahrenheit', 'feels_like_celsius', 'feels_like_fahrenheit']
df = df.drop(columns=cols_vazamento, errors='ignore')

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

print("Tratando valores nulos...")
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("Aplicando detecção de anomalias (MAD)...")
outliers_mask = pd.DataFrame()
for col in numeric_cols:
    outliers_mask[col] = detect_outliers_mad(df[col])

# Cria uma feature nova: 'has_anomaly' (1 se tiver pelo menos 1 outlier nas colunas numéricas, 0 caso contrário)
df['has_anomaly'] = outliers_mask.any(axis=1).astype(int)
print(f"Total de registros sinalizados como anômalos: {df['has_anomaly'].sum()}")

# 4. Normalização (RobustScaler)
# Ignoramos as colunas de data/hora epoch e a nossa nova flag de anomalia na hora de escalar
cols_to_scale = [col for col in numeric_cols if col not in ['last_updated_epoch', 'has_anomaly']]

print("Normalizando os dados com RobustScaler...")
scaler = RobustScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# 5. Exportação do Dataset Processado
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'Processed_Weather_Data.csv')

df.to_csv(output_file, index=False)
print(f"Sucesso! Dataset tratado e normalizado salvo em: {output_file}")