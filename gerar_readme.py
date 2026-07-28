conteudo = """# 🤖 RL FX Trader: MetaTrader 5 PPO Trading Bot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-black)
![Stable-Baselines3](https://img.shields.io/badge/Stable%20Baselines-3-green)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-orange)

Um ecossistema completo de Algorithmic Trading baseado em Aprendizado por Reforço (Reinforcement Learning) utilizando o algoritmo **PPO (Proximal Policy Optimization)**. O bot foi desenhado para operar no mercado de Forex (padrão EURUSD), atuando como um *Scalper/Day Trader* no tempo gráfico de 5 minutos (M5), com consciência de tendências macro (H1 e D1).

## ✨ Principais Funcionalidades

* **Integração Nativa com MT5:** Coleta de dados históricos e envio de ordens ao vivo diretamente via biblioteca oficial `MetaTrader5`.
* **Prevenção de Lookahead Bias:** Arquitetura multitimeframe (M5 + H1 + D1) sincronizada com técnica de `shift(1)`, garantindo que o modelo aprenda apenas com dados já fechados.
* **Ambiente Customizado (Gymnasium):**
  * **3 Ações Discretas:** Comprar (Long), Vender (Short) e Hold (Ficar de fora).
  * **Alvos Físicos (Pips):** Stop Loss dinâmico de 3 pips e Take Profit de 5 pips.
  * **Time-Out:** Encerramento automático de operações paradas após 30 minutos (6 candles).
  * **Custos Reais:** Penalização de spread e taxas (0.5 pip por trade) para evitar *overtrading*.
* **Validação Transparente:** Script de backtest com métricas de Taxa de Acerto, Quantidade de Holds e Retorno Acumulado.

## 📂 Estrutura do Projeto

| Arquivo | Descrição |
| :--- | :--- |
| `dados.py` | Conecta ao MT5, baixa os candles (M5), garante sincronia UTC e divide os dados temporalmente (Treino/Validação/Teste). |
| `indicadores.py` | Calcula RSI e médias móveis, gera features relativas e sincroniza tendências macro (H1, D1) com o M5. |
| `ambiente.py` | Simulador das regras do mercado criado com `gymnasium.Env`. |
| `treinar.py` | Pipeline de treinamento do agente usando `MlpPolicy` do Stable-Baselines3 (150.000 steps). |
| `validar.py` | Script de avaliação de performance do modelo treinado em dados invisíveis (Out-of-Sample). |
| `live.py` | Operador ao vivo. Lê o mercado em tempo real, consulta a IA e despacha as ordens para o MT5 via número mágico. |

## 🚀 Como Começar

### Pré-requisitos
1. **MetaTrader 5** instalado, logado em uma conta (preferencialmente Demo) e com a opção *"Permitir negociação automatizada (Algorithmic trading)"* ativada.
2. Python 3.9 ou superior.

### Instalação

1. Clone o repositório:
```bash
git clone [https://github.com/Renato-Bapxtista/robo_trade_scalp_1.0.git](https://github.com/Renato-Bapxtista/robo_trade_scalp_1.0.git)
cd robo_trade_scalp_1.0