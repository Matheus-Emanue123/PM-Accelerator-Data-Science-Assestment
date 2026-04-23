import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / 'input' / 'GlobalWeatherRepository.csv'
OUTPUT_DIR = ROOT_DIR / 'output'

def detect_outliers_mad(series, threshold=3.5):
    """Retorna uma máscara booleana indicando outliers baseados no MAD."""
    mediana = series.median()
    mad = np.median(np.abs(series - mediana))
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z_score = 0.6745 * (series - mediana) / mad
    return np.abs(modified_z_score) > threshold

def load_and_process_data(file_path=None):
    """Lê o dataset bruto, limpa, normaliza e retorna o DataFrame na memória."""
    # print("Iniciando processamento de dados na memória...")
    if file_path is None:
        file_path = INPUT_FILE
    df = pd.read_csv(file_path)

    cols_vazamento = ['temperature_fahrenheit', 'feels_like_celsius', 'feels_like_fahrenheit']
    df = df.drop(columns=cols_vazamento, errors='ignore')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # print("Tratando valores nulos...")
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # print("Aplicando detecção de anomalias (MAD)...")
    outliers_mask = pd.DataFrame()
    for col in numeric_cols:
        outliers_mask[col] = detect_outliers_mad(df[col])

    df['has_anomaly'] = outliers_mask.any(axis=1).astype(int)
    # print(f"Total de registros sinalizados como anômalos: {df['has_anomaly'].sum()}")

    cols_to_scale = [col for col in numeric_cols if col not in ['last_updated_epoch', 'has_anomaly']]
    
    # print("Normalizando os dados com RobustScaler...")
    scaler = RobustScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # print("Processamento concluído! DataFrame pronto para uso.\n")
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / 'Processed_Weather_Data.csv', index=False)
    return df

if __name__ == "__main__":
    df_processado = load_and_process_data()
    # print("Tamanho do DataFrame processado:", df_processado.shape)