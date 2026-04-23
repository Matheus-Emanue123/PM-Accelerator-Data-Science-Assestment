import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / 'input' / 'GlobalWeatherRepository.csv'
OUTPUT_DIR = ROOT_DIR / 'output'

df = pd.read_csv(INPUT_FILE)
df = df.drop(columns=['temperature_fahrenheit', 'feels_like_celsius', 'feels_like_fahrenheit'], errors='ignore')

df['last_updated'] = pd.to_datetime(df['last_updated'])

df['year_month'] = df['last_updated'].dt.to_period('M')

sns.set_theme(style="whitegrid")

df['Region'] = df['timezone'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'Other')

fig, ax = plt.subplots(figsize=(14, 6))


sns.boxplot(data=df, x='Region', y='temperature_celsius', palette='Set3', ax=ax)
ax.set_title('Climate Analysis: Variação Térmica por Região Global', fontsize=16, fontweight='bold')
ax.set_xlabel('Região Geográfica')
ax.set_ylabel('Temperatura (°C)')

plt.xticks(rotation=45)
plt.tight_layout()
OUTPUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_DIR / '06_climate_analysis.png', dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Environmental Impact: Clima vs. Qualidade do Ar', fontsize=16, fontweight='bold')

limite_poluicao = df['air_quality_PM2.5'].quantile(0.95)

df_filtrado = df[(df['air_quality_PM2.5'] < limite_poluicao) & (df['air_quality_PM2.5'] >= 0)]

sns.regplot(data=df_filtrado, x='wind_kph', y='air_quality_PM2.5', scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'}, ax=axes[0])
axes[0].set_title('Impacto do Vento na Dispersão de Partículas (PM2.5)')
axes[0].set_xlabel('Velocidade do Vento (km/h)')
axes[0].set_ylabel('Concentração PM2.5 (Poluição)')

sns.scatterplot(data=df_filtrado, x='temperature_celsius', y='air_quality_Ozone', hue='uv_index', palette='flare', alpha=0.6, s=15, ax=axes[1])
axes[1].set_title('Relação entre Calor, Índice UV e Formação de Ozônio (O3)')
axes[1].set_xlabel('Temperatura (°C)')
axes[1].set_ylabel('Nível de Ozônio (O3)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUTPUT_DIR / '07_environmental_impact.png', dpi=300)
plt.close()

# print("Unique Analyses concluídas! Painéis '06' e '07' salvos.")