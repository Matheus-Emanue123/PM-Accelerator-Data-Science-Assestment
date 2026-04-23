import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression, mutual_info_regression

from data_processing import load_and_process_data

df = load_and_process_data()

target_col = 'temperature_celsius'

numeric_df = df.select_dtypes(include=[np.number])

X = numeric_df.drop(columns=[target_col])
y = numeric_df[target_col]

print(f"Avaliando {X.shape[1]} features numéricas contra o target '{target_col}'...\n")

print("Calculando F-Regression...")
f_scores, p_values = f_regression(X, y)
f_scores_normalized = f_scores / np.max(f_scores) 
f_reg_results = pd.Series(f_scores_normalized, index=X.columns).sort_values(ascending=False)

print("Calculando Mutual Information (isso pode levar alguns segundos)...")
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_scores_normalized = mi_scores / np.max(mi_scores)
mi_results = pd.Series(mi_scores_normalized, index=X.columns).sort_values(ascending=False)

top_n = 15

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Importance: F-Regression vs Mutual Information', fontsize=16, fontweight='bold')

sns.barplot(
    x=f_reg_results.head(top_n).values, 
    y=f_reg_results.head(top_n).index, 
    ax=axes[0], 
    hue=f_reg_results.head(top_n).index, 
    palette='viridis', 
    legend=False
)
axes[0].set_title('Top 15 Features (F-Regression - Linear)', fontsize=14)

sns.barplot(
    x=mi_results.head(top_n).values, 
    y=mi_results.head(top_n).index, 
    ax=axes[1], 
    hue=mi_results.head(top_n).index, 
    palette='magma', 
    legend=False
)
axes[1].set_title('Top 15 Features (Mutual Information - Não Linear)', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('output/feature_importance_comparison.png', dpi=300)
plt.show()

features_finais = [
    'temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 
    'precip_mm', 'cloud', 'visibility_km', 'uv_index'
]

corr_matrix = df[features_finais].corr(method='pearson')

fig, ax = plt.subplots(figsize=(12, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 10})
            
ax.set_title("Matriz de Correlação de Pearson (Features Refinadas)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig('output/correlation_heatmap.png', dpi=300)
plt.show()

print("Análise concluída e gráfico salvo!")