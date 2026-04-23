import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detect_outliers_mad(series, threshold=3.5):
    mediana = series.median()
    mad = np.median(np.abs(series - mediana))
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z_score = 0.6745 * (series - mediana) / mad
    return np.abs(modified_z_score) > threshold

print("Carregando dados para EDA...")
df_eda = pd.read_csv('input/GlobalWeatherRepository.csv')

df_eda = df_eda.drop(columns=['temperature_fahrenheit', 'feels_like_celsius', 'feels_like_fahrenheit'], errors='ignore')
numeric_cols = df_eda.select_dtypes(include=[np.number]).columns

outliers_mask = pd.DataFrame()
for col in numeric_cols:
    outliers_mask[col] = detect_outliers_mad(df_eda[col])
df_eda['is_anomaly'] = outliers_mask.any(axis=1).map({True: 'Anomalia (Outlier)', False: 'Normal'})

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Distribuição Global de Temperatura e Precipitação', fontsize=16, fontweight='bold')

# Histograma de Temperatura
sns.histplot(df_eda['temperature_celsius'], bins=50, kde=True, color='coral', ax=axes[0])
axes[0].set_title('Distribuição da Temperatura (°C)')
axes[0].set_xlabel('Temperatura (°C)')
axes[0].set_ylabel('Frequência (Cidades)')

# Histograma de Precipitação (usando log scale porque a maioria dos dias é 0mm)
sns.histplot(df_eda['precip_mm'], bins=50, color='royalblue', ax=axes[1])
axes[1].set_yscale('log')
axes[1].set_title('Distribuição da Precipitação (mm) - Escala Logarítmica')
axes[1].set_xlabel('Precipitação (mm)')
axes[1].set_ylabel('Frequência (Log)')

plt.tight_layout()
plt.savefig('output/01_temp_precip_dist.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df_eda, 
    x='temperature_celsius', 
    y='humidity', 
    hue='is_anomaly', 
    palette={'Normal': '#2ecc71', 'Anomalia (Outlier)': '#e74c3c'},
    alpha=0.6,
    s=20
)
ax.set_title('Análise de Anomalias Climáticas (Temperatura vs Umidade)', fontsize=14, fontweight='bold')
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Umidade (%)')
plt.tight_layout()
plt.savefig('output/02_anomaly_analysis.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(14, 7))

scatter = ax.scatter(
    df_eda['longitude'], 
    df_eda['latitude'], 
    c=df_eda['temperature_celsius'], 
    cmap='coolwarm', 
    alpha=0.8,
    s=10
)
plt.colorbar(scatter, label='Temperatura (°C)')
ax.set_title('Padrões Geográficos Globais de Temperatura', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig('output/03_geographical_patterns.png', dpi=300)
plt.close()

print("EDA concluído! As 3 imagens foram salvas na raiz do projeto.")