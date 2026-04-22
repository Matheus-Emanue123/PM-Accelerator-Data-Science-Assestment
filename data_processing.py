import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer

file_path = 'input/GlobalWeatherRepository.csv'
print(f"Carregando dados de: {file_path}")
df = pd.read_csv(file_path)

target_col = 'temperature_celsius'

leakage_cols = ['temperature_fahrenheit', 'feels_like_celsius', 'feels_like_fahrenheit']
df_cleaned = df.drop(columns=leakage_cols, errors='ignore')

numeric_df = df_cleaned.select_dtypes(include=[np.number])

imputer = SimpleImputer(strategy='median')
numeric_data_imputed = imputer.fit_transform(numeric_df)
numeric_df = pd.DataFrame(numeric_data_imputed, columns=numeric_df.columns)

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

sns.barplot(
    x=mi_results.head(top_n).values, 
    y=mi_results.head(top_n).index, 
    ax=axes[1], 
    hue=mi_results.head(top_n).index, 
    palette='magma', 
    legend=False
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('feature_importance_comparison.png', dpi=300)
plt.show()

