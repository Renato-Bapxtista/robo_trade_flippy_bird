# 🐦 Robo Trade: Flippy Bird (RL PPO)

Um bot de Scalping e Day Trade impulsionado por Inteligência Artificial e Aprendizado por Reforço (*Reinforcement Learning*). O agente utiliza o algoritmo **PPO (Proximal Policy Optimization)** da biblioteca `Stable-Baselines3` para operar no MetaTrader 5 (MT5).

O grande diferencial deste projeto é a sua lógica de gerenciamento de risco e recompensas, inspirada na mecânica do jogo **Flappy Bird**.

---

## 🎮 A Mecânica "Flippy Bird"

Em vez de usar alvos fixos tradicionais, o robô navega pelo gráfico surfando em um canal dinâmico:
- **O Voo:** Quando o robô entra em uma operação (Compra ou Venda), um canal se forma com um *Stop Loss* (chão) e um *Take Profit* (teto).
- **Trailing Dinâmico:** Conforme o preço avança a favor, o teto (TP) é empurrado para frente, e o chão (SL) é puxado para cima, encurralando o lucro.
- **Recompensas Densas (As Moedinhas):** Para ensinar o robô a manter o trade vivo, a IA recebe uma pequena pontuação positiva (`+0.0001`) a cada "cano" que ela ultrapassa — ou seja, toda vez que consegue arrastar o *Trailing Stop* para uma área de lucro, recompensando a sobrevivência!

## ⏳ Sistema de Fases e Vidas (Game Over Parcial)

O mercado muda a cada hora, então o robô trata cada hora cheia (H1) como uma **nova fase do jogo**.
- **Limite de Vidas:** O robô possui um limite de "mortes" (Stops no prejuízo) por hora.
- **Punição/Hold:** Se o limite de stops for estourado dentro da mesma hora, o robô entra em *Game Over Parcial* e é forçado a ficar de fora (**HOLD**) até que o relógio vire e uma nova fase comece, zerando suas vidas. Isso evita *overtrading* em mercados laterais ou com muito ruído.

## 🧠 IA Preditiva Hierárquica (A Bola de Cristal)

O ecossistema conta com dois modelos de IA trabalhando juntos:
1. **Previsor H1 (Random Forest):** Antes do candle de 1 hora abrir, modelos leves estudam o contexto macro e prevêem a **Direção**, o **Tamanho** e a **Distância para a média diária** do próximo candle de H1.
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