# 🐦 Robo Trade: Flippy Bird (RL PPO)

Um bot de Scalping e Day Trade impulsionado por Inteligência Artificial e Aprendizado por Reforço (*Reinforcement Learning*). O agente utiliza o algoritmo **PPO (Proximal Policy Optimization)** da biblioteca `Stable-Baselines3` para operar no MetaTrader 5 (MT5).

O grande diferencial deste projeto é a sua lógica de gerenciamento de risco e recompensas, inspirada na mecânica do jogo **Flappy Bird**.

---

## 🎮 A Mecânica "Flippy Bird"

Em vez de usar alvos fixos tradicionais, o robô navega pelo gráfico surfando em um canal dinâmico:

- **O Voo:** Quando o robô entra em uma operação (Compra ou Venda), um canal se forma com um *Stop Loss* (chão) e um *Take Profit* (teto).

- **Trailing Dinâmico:** Conforme o preço avança a favor, o teto (TP) é empurrado para frente, e o chão (SL) é puxado para cima, encurralando o lucro.
- **Recompensas Densas (As Moedinhas):** Para ensinar o robô a manter o trade vivo, a IA recebe uma pequena pontuação positiva (`+0.0001`) a cada "cano" que ela ultrapassa — ou seja, toda vez que consegue arrastar o *Trailing Stop* para uma área de lucro, recompensando a sobrevivência.
- **Virada de Mão (Flip):** O passarinho é ágil! Se o robô estiver comprado e perceber que o mercado vai cair, ele tem autonomia para encerrar a compra instantaneamente e já abrir uma operação de venda (e vice-versa). Isso permite que ele surfe a nova tendência sem precisar esperar bater no Stop Loss!

## ⏳ Sistema de Fases e Vidas (Game Over Parcial)

O mercado muda a cada hora, então o robô trata cada hora cheia (H1) como uma **nova fase do jogo**.

- **Limite de Vidas:** O robô possui um limite de "mortes" (Stops no prejuízo) por hora.
- **Punição/Hold:** Se o limite de stops for estourado dentro da mesma hora, o robô entra em *Game Over Parcial* e é forçado a ficar de fora (**HOLD**) até que o relógio vire e uma nova fase comece, zerando suas vidas. Isso evita *overtrading* em mercados laterais ou com muito ruído.

## 🧠 IA Preditiva Hierárquica (A Bola de Cristal)

O ecossistema conta com dois modelos de IA trabalhando juntos:

1. **Previsor H1 (Random Forest):** Antes do candle de 1 hora abrir, modelos leves estudam o contexto macro e prevêem a **Direção**, o 

**Tamanho** e a **Distância para a média diária** do próximo candle de H1.
2. **Agente M5 (PPO):** O robô principal lê as previsões do H1 no início da fase e toma as decisões operacionais (Compra, Venda ou Hold) a cada 5 minutos, executando a mecânica do *Flippy Bird*.

---

## 🛠️ Tecnologias Utilizadas

- `Python 3.10+`
- `MetaTrader5` (Integração e extração de dados)
- `Stable-Baselines3` (Treinamento do agente PPO)
- `Gymnasium` (Criação do ambiente do jogo financeiro)
- `Scikit-Learn` (IA preditiva Random Forest para H1)
- `Pandas & NumPy` (Engenharia de features)

---

## 🚀 Como Executar

### 1. Instalar as dependências

```bash
pip install pandas numpy scikit-learn stable-baselines3 gymnasium MetaTrader5

```

### 2. Treinar o Robô

Extraia os dados do MT5, treine a IA preditiva e o agente PPO executando:

```bash
python treinar.py
(O modelo será salvo como robo_financeiro_ppo.zip)
```

### 3. Validar a Estratégia

Faça o backtest em dados invisíveis para testar a paciência (Holds) e a taxa de acerto do passarinho:

```bash
python validar.py
```

### 4. Modo Live (Conta Demo/Real)

Conecte ao MT5 aberto e deixe o robô operar em tempo real:

```bash
python operar_live.py
```

Aviso: Este é um projeto de estudo sobre Aprendizado por Reforço no mercado financeiro. Não utilize em conta real sem testes exaustivos. O autor não se responsabiliza por perdas financeiras.

#### 📊 Relatório de Treinamento e Validação - Flippy Bird Bot

## 🚀 Último Treinamento (1 Milhão de Timesteps)

- **Modelo Salvo:** `robo_financeiro_ppo.zip`
- **Data/Hora:** Março de 2026

## 📈 Resultado da Validação (Dados Invisíveis)

```text
================ RELATÓRIO DE PERFORMANCE (3 AÇÕES) ================
Decisões de Análise Técnica:
 ⏸️ Quantidade de vezes em HOLD (Ficou de Fora): 6483
 🟢 Ordens de COMPRA (Long): 0
 🔴 Ordens de VENDA (Short): 5
---------------------------------------------------------------------
Total de Trades Efetivamente Fechados: 5
 🔥 Operações com LUCRO (TP/Trailing): 0
 💀 Operações com PREJUÍZO (SL): 3
 🎯 Taxa de Acerto Real dos Trades: 0.0%
---------------------------------------------------------------------
 💰 RETORNO ACUMULADO FINAL DO ROBÔ: -0.096%
=====================================================================
