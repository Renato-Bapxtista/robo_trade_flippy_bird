from datetime import datetime, timezone
import numpy as np
from stable_baselines3 import PPO
from dados import obter_dados_mt5, separar_dados_temporal
from indicadores import preparar_dados_mercado
from ambiente import AmbienteTrading

def executar_validacao():
    print("--- 1. Carregando Dados Invisíveis ---")
    df_bruto = obter_dados_mt5(ativo="EURUSD", inicio=datetime(2026, 1, 1, tzinfo=timezone.utc))
    df_com_features = preparar_dados_mercado(df_bruto)
    
    # Separando os dados de teste real
    _, _, df_teste = separar_dados_temporal(df_com_features)
    print(f"Quantidade de candles para o teste real: {len(df_teste)}")
    
    # Inicializa o ambiente com a nova regra de 3 ações
    env_teste = AmbienteTrading(df_teste)
    
    print("\n--- 2. Carregando Cérebro do Robô ---")
    model = PPO.load("robo_financeiro_ppo")
    
    print("\n--- 3. Iniciando Simulação de Operações ---")
    obs, info = env_teste.reset()
    finalizado = False
    
    historico_recompensas = []
    total_vendas = 0   # Ação 0
    total_hold = 0     # Ação 1
    total_compras = 0    # Ação 2
    
    while not finalizado:
        # Predição determinística pura (sem aleatoriedade)
        action, _states = model.predict(obs, deterministic=True)
        
        # Rastreia qual botão a IA apertou
        if action == 2:
            total_compras += 1
        elif action == 0:
            total_vendas += 1
        else:
            total_hold += 1
            
        obs, recompensa, finalizado, truncado, info = env_teste.step(action)
        
        # Guardamos a recompensa apenas se o robô REALMENTE operou (Comprou ou Vendeu)
        # Se foi Hold, a recompensa é zero e não conta como trade fechado
        if action == 0 or action == 2:
            historico_recompensas.append(recompensa)
        
    # --- 4. Métricas Avançadas de Performance ---
    historico_recompensas = np.array(historico_recompensas)
    retorno_total = np.sum(historico_recompensas) * 100
    
    operacoes_ganhas = np.sum(historico_recompensas > 0)
    operacoes_perdidas = np.sum(historico_recompensas < 0)
    total_trades = len(historico_recompensas)
    
    taxa_acerto = (operacoes_ganhas / total_trades) * 100 if total_trades > 0 else 0
    
    print("\n================ RELATÓRIO DE PERFORMANCE (3 AÇÕES) ================")
    print(f"Decisões de Análise Técnica:")
    print(f" ⏸️ Quantidade de vezes em HOLD (Ficou de Fora): {total_hold}")
    print(f" 🟢 Ordens de COMPRA (Long): {total_compras}")
    print(f" 🔴 Ordens de VENDA (Short): {total_vendas}")
    print(f"---------------------------------------------------------------------")
    print(f"Total de Trades Efetivamente Fechados: {total_trades}")
    print(f" 🔥 Operações com LUCRO (TP/Trailing): {operacoes_ganhas}")
    print(f" 💀 Operações com PREJUÍZO (SL): {operacoes_perdidas}")
    print(f" 🎯 Taxa de Acerto Real dos Trades: {taxa_acerto:.2f}%")
    print(f"---------------------------------------------------------------------")
    print(f" 💰 RETORNO ACUMULADO FINAL DO ROBÔ: {retorno_total:.3f}%")
    print("=====================================================================")

if __name__ == "__main__":
    executar_validacao()
