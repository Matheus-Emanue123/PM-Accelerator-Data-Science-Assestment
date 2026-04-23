import subprocess
import sys
import time

def run_script(script_name):

    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ETAPA: {script_name}")
    print(f"{'='*60}")
    
    time.sleep(1)
    
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ SUCESSO: {script_name} finalizado sem erros.")
    except subprocess.CalledProcessError:
        print(f"\n❌ FALHA CRÍTICA: Erro detectado ao executar {script_name}.")
        print("Abortando o resto do pipeline para evitar corrupção de dados.")
        sys.exit(1) 

if __name__ == "__main__":
    print("🌟 INICIANDO PIPELINE DE MACHINE LEARNING (WEATHER FORECAST) 🌟")
    print("Iniciando orquestração de dados, análises e modelagem...\n")
    
    pipeline = [
        "data_processing.py",    
        "eda_visualization.py",  
        "feature_analysis.py", 
        "model_forecast.py",     
        "unique_analyses.py"      
    ]

    tempo_inicio = time.time()

    for script in pipeline:
        run_script(script)

    tempo_fim = time.time()
    tempo_total = tempo_fim - tempo_inicio

    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE CONCLUÍDO COM SUCESSO! 🎉")
    print(f"⏳ Tempo total de execução: {tempo_total:.2f} segundos.")
    print("Todos os dados processados e gráficos (01 a 07) foram gerados na raiz do projeto.")
    print(f"{'='*60}\n")