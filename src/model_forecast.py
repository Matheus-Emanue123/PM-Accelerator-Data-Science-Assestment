import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / 'output'

print("Carregando dados processados...")
df = pd.read_csv(OUTPUT_DIR / 'Processed_Weather_Data.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])
df = df.sort_values(by='last_updated')

features_finais = [
    'humidity', 'wind_kph', 'pressure_mb', 
    'precip_mm', 'cloud', 'visibility_km', 'uv_index'
]
target = 'temperature_celsius'

X = df[features_finais]
y = df[target]

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Dados divididos: Treino ({len(X_train)}), Teste ({len(X_test)}).\n")

resultados = {}

def avaliar_modelo(modelo, nome, fit=True):
    if fit:
        modelo.fit(X_train, y_train)
    prev = modelo.predict(X_test)
    r2 = r2_score(y_test, prev)
    mae = mean_absolute_error(y_test, prev)
    resultados[nome] = {'R2': r2, 'MAE': mae}
    print(f"{nome.ljust(35)} | R2: {r2:.4f} | MAE: {mae:.2f}°C")
    return modelo

print("--- FASE 1: Modelos Individuais (Default) ---")
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

avaliar_modelo(lr, "1. Regressão Linear")
avaliar_modelo(rf, "2. Random Forest (Default)")
avaliar_modelo(xgb, "3. XGBoost (Default)")

print("\n--- FASE 2: Ensembles Iniciais ---")
ens1 = VotingRegressor(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)])
avaliar_modelo(ens1, "4. Ensemble 1 (LR + RF + XGB)")

ens2 = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb)])
avaliar_modelo(ens2, "5. Ensemble 2 (Sem LR)")

print("\n--- FASE 3: Grid Search (Isso leva um minutinho) ---")

grid_rf = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), 
                       {'n_estimators': [100, 200], 'max_depth': [10, 15]}, 
                       cv=3, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
rf_opt = grid_rf.best_estimator_
avaliar_modelo(rf_opt, "6. Random Forest (Otimizado)", fit=False) 

grid_xgb = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1), 
                        {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 7]}, 
                        cv=3, scoring='r2', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
xgb_opt = grid_xgb.best_estimator_
avaliar_modelo(xgb_opt, "7. XGBoost (Otimizado)", fit=False)

print("\n--- FASE 4: O Ensemble Definitivo ---")
ens3 = VotingRegressor(estimators=[('rf_opt', rf_opt), ('xgb_opt', xgb_opt)])
avaliar_modelo(ens3, "8. Ensemble 3 (Otimizados)")

nomes = list(resultados.keys())
r2_valores = [resultados[n]['R2'] for n in nomes]

cores = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#3498db', '#9b59b6', '#2ecc71', '#27ae60']

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(nomes, r2_valores, color=cores)

for bar in bars:
    yval = bar.get_height()
    offset = 0.01 if yval >= 0 else -0.05
    ax.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_title('Comparativo Completo: Modelos Individuais vs Ensembles (R² Score)', fontsize=16, fontweight='bold')
ax.set_ylabel('R² Score (Acurácia)')

ax.set_xticklabels(nomes, rotation=45, ha='right', fontsize=11)

ax.axhline(0, color='black', linewidth=1)

plt.tight_layout()
OUTPUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_DIR / '05_evolucao_completa_modelos.png', dpi=300)
plt.show()

print("\nRelatório gerado com sucesso! Gráfico salvo com as 8 barras.")