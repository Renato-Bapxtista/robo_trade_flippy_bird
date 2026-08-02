import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd

class AmbienteTrading(gymnasium.Env):
    def __init__(self, df: pd.DataFrame, sl_pips: float = 0.0005, tp_pips: float = 0.0010, max_episode_steps: int | None = None):
        super(AmbienteTrading, self).__init__()
        self.pips_respiro_trailing = 0.0005  # Equivalente a 5 pips no EURUSD
        self.df = df.reset_index(drop=True)
        self.colunas_features = [col for col in df.columns if col.startswith("feat_")]
        self.num_features = len(self.colunas_features)
        
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.custo_transacao = 0.0003
        
        self.action_space = spaces.Discrete(3)  # 0: Venda, 1: Hold, 2: Compra
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.num_features,), dtype=np.float32
        )
        
        self.passo_atual = 0
        
        # Variáveis de controle do "Jogo" (Fases por Hora)
        self.hora_atual_bloco = None
        self.mortes_na_hora = 0       # "Mortes" = Stop Loss atingido
        self.limite_mortes_hora = 10  # Exemplo: Máximo de 10 stops por fase/hora
        
        # Estado da Posição Atual
        self.posicao_aberta = None    # None, "COMPRA" ou "VENDA"
        self.preco_entrada = 0.0
        self.alvo_sl = 0.0
        self.alvo_tp = 0.0
        self.duracao_trade = 0
        self.max_duracao = 12         # Exemplo: 12 candles = 1 hora de trade máximo
        # Controle de steps por episódio (limita comprimento do episódio)
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

    def _pegar_observacao(self):
        return np.array(self.df.iloc[self.passo_atual][self.colunas_features].values, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.passo_atual = 0
        self.mortes_na_hora = 0
        self.hora_atual_bloco = None
        self.posicao_aberta = None
        self._step_count = 0
        return self._pegar_observacao(), {}

    def step(self, action):
        row_atual = self.df.iloc[self.passo_atual]
        timestamp_atual = row_atual["time"]
        hora_cheia_atual = timestamp_atual.floor("h")
        
        # --- TROCA DE FASE (Virada de Hora H1) ---
        if self.hora_atual_bloco != hora_cheia_atual:
            self.hora_atual_bloco = hora_cheia_atual
            self.mortes_na_hora = 0  # Reseta as "vidas" para a nova fase!

        recompensa = 0.0
        fechou_trade_neste_passo = False
        
        preco_open = row_atual["open"]
        preco_high = row_atual["high"]
        preco_low = row_atual["low"]
        preco_close = row_atual["close"]

        # =====================================================================
        # 1. VIRADA DE MÃO (FLIP) - O Agente decide trocar de direção no ar!
        # =====================================================================
        if self.posicao_aberta == "COMPRA" and action == 0:  # Estava comprado e mandou VENDER
            lucro_flip = (preco_open - self.preco_entrada) / self.preco_entrada
            recompensa += lucro_flip
            if lucro_flip < 0:
                self.mortes_na_hora += 1
            
            # Vira a mão para VENDA imediatamente (se não tiver morrido de vez na fase)
            if self.mortes_na_hora < self.limite_mortes_hora:
                self.posicao_aberta = "VENDA"
                self.preco_entrada = preco_open
                self.alvo_sl = self.preco_entrada + self.sl_pips
                self.alvo_tp = self.preco_entrada - self.tp_pips
                self.duracao_trade = 0
                recompensa -= (self.custo_transacao / self.preco_entrada)
            else:
                self.posicao_aberta = None
                fechou_trade_neste_passo = True

        elif self.posicao_aberta == "VENDA" and action == 2:  # Estava vendido e mandou COMPRAR
            lucro_flip = (self.preco_entrada - preco_open) / self.preco_entrada
            recompensa += lucro_flip
            if lucro_flip < 0:
                self.mortes_na_hora += 1
                
            # Vira a mão para COMPRA imediatamente
            if self.mortes_na_hora < self.limite_mortes_hora:
                self.posicao_aberta = "COMPRA"
                self.preco_entrada = preco_open
                self.alvo_sl = self.preco_entrada - self.sl_pips
                self.alvo_tp = self.preco_entrada + self.tp_pips
                self.duracao_trade = 0
                recompensa -= (self.custo_transacao / self.preco_entrada)
            else:
                self.posicao_aberta = None
                fechou_trade_neste_passo = True

        # =====================================================================
        # 2. GERENCIAMENTO DE POSIÇÃO ABERTA (Trailing e Moedinhas do Flippy Bird)
        # =====================================================================
        if self.posicao_aberta is not None:
            self.duracao_trade += 1
            
            if self.posicao_aberta == "COMPRA":
                if preco_high > (self.preco_entrada + self.pips_respiro_trailing):
                    delta = preco_high - self.preco_entrada
                    novo_sl = (self.preco_entrada - self.sl_pips) + delta
                    if novo_sl > self.alvo_sl:
                        recompensa += 0.00015  # Moedinha!
                        self.alvo_sl = novo_sl
                        self.alvo_tp = preco_high + self.tp_pips
                
                if preco_low <= self.alvo_sl:
                    recompensa += (self.alvo_sl - self.preco_entrada) / self.preco_entrada
                    if (self.alvo_sl - self.preco_entrada) < 0:
                        self.mortes_na_hora += 1
                    self.posicao_aberta = None
                    fechou_trade_neste_passo = True
                elif preco_high >= self.alvo_tp:
                    recompensa += (self.alvo_tp - self.preco_entrada) / self.preco_entrada
                    self.posicao_aberta = None
                    fechou_trade_neste_passo = True
                    
            elif self.posicao_aberta == "VENDA":
                if preco_low < (self.preco_entrada - self.pips_respiro_trailing):
                    delta = self.preco_entrada - preco_low
                    novo_sl = (self.preco_entrada + self.sl_pips) - delta
                    if novo_sl < self.alvo_sl:
                        recompensa += 0.00015  # Moedinha!
                        self.alvo_sl = novo_sl
                        self.alvo_tp = preco_low - self.tp_pips

                if preco_high >= self.alvo_sl:
                    recompensa += (self.preco_entrada - self.alvo_sl) / self.preco_entrada
                    if (self.preco_entrada - self.alvo_sl) < 0:
                        self.mortes_na_hora += 1
                    self.posicao_aberta = None
                    fechou_trade_neste_passo = True
                elif preco_low <= self.alvo_tp:
                    recompensa += (self.preco_entrada - self.alvo_tp) / self.preco_entrada
                    self.posicao_aberta = None
                    fechou_trade_neste_passo = True

        # =====================================================================
        # 3. ABERTURA DE NOVA POSIÇÃO (Se ele estava de fora e decidiu entrar)
        # =====================================================================
        if self.posicao_aberta is None and not fechou_trade_neste_passo:
            if self.mortes_na_hora >= self.limite_mortes_hora:
                action = 1  # Força Hold se estourou o limite de stops na fase
            
            if action == 0:  # VENDA
                self.posicao_aberta = "VENDA"
                self.preco_entrada = preco_close
                self.alvo_sl = self.preco_entrada + self.sl_pips
                self.alvo_tp = self.preco_entrada - self.tp_pips
                self.duracao_trade = 0
                recompensa -= (self.custo_transacao / self.preco_entrada)
                
            elif action == 2:  # COMPRA
                self.posicao_aberta = "COMPRA"
                self.preco_entrada = preco_close
                self.alvo_sl = self.preco_entrada - self.sl_pips
                self.alvo_tp = self.preco_entrada + self.tp_pips
                self.duracao_trade = 0
                recompensa -= (self.custo_transacao / self.preco_entrada)

        # Avança para o próximo candle
        self.passo_atual += 1
        self._step_count += 1
        finalizado = self.passo_atual >= len(self.df) - 1 or (self.max_episode_steps is not None and self._step_count >= self.max_episode_steps)
        nova_obs = self._pegar_observacao() if not finalizado else np.zeros((self.num_features,), dtype=np.float32)
        
        return nova_obs, recompensa, finalizado, False, {}