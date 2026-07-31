import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd

class AmbienteTrading(gymnasium.Env):
    def __init__(self, df: pd.DataFrame, sl_pips: float = 0.0005, tp_pips: float = 0.0010):
        super(AmbienteTrading, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.colunas_features = [col for col in df.columns if col.startswith("feat_")]
        self.num_features = len(self.colunas_features)
        
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.custo_transacao = 0.00005
        
        self.action_space = spaces.Discrete(3)  # 0: Venda, 1: Hold, 2: Compra
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.num_features,), dtype=np.float32
        )
        
        self.passo_atual = 0
        
        # Variáveis de controle do "Jogo" (Fases por Hora)
        self.hora_atual_bloco = None
        self.mortes_na_hora = 0       # "Mortes" = Stop Loss atingido
        self.limite_mortes_hora = 3   # Exemplo: Máximo de 3 stops por fase/hora
        
        # Estado da Posição Atual
        self.posicao_aberta = None    # None, "COMPRA" ou "VENDA"
        self.preco_entrada = 0.0
        self.alvo_sl = 0.0
        self.alvo_tp = 0.0
        self.duracao_trade = 0
        self.max_duracao = 12         # Exemplo: 12 candles = 1 hora de trade máximo

    def _pegar_observacao(self):
        return np.array(self.df.iloc[self.passo_atual][self.colunas_features].values, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.passo_atual = 0
        self.mortes_na_hora = 0
        self.hora_atual_bloco = None
        self.posicao_aberta = None
        return self._pegar_observacao(), {}

    def step(self, action):
        row_atual = self.df.iloc[self.passo_atual]
        timestamp_atual = row_atual["time"]
        hora_cheia_atual = timestamp_atual.floor("h")
        
        # --- TROCA DE FASE (Virada de Hora H1) ---
        if self.hora_atual_bloco != hora_cheia_atual:
            self.hora_atual_bloco = hora_cheia_atual
            self.mortes_na_hora = 0  # Reseta as "vidas/mortes" para a nova fase!

        recompensa = 0.0
        fechou_trade_neste_passo = False

        # =====================================================================
        # 1. GERENCIAMENTO DE POSIÇÃO ABERTA (O Voo do Passarinho)
        # =====================================================================
        if self.posicao_aberta is not None:
            self.duracao_trade += 1
            
            # Aqui definimos as variáveis para o Python não reclamar!
            preco_high = row_atual["high"]
            preco_low = row_atual["low"]
            #preco_close = row_atual["close"]

            if self.posicao_aberta == "COMPRA":
                # Trailing Dinâmico Proporcional (Só sobe se der um respiro de 5 pips)
                if preco_high > (self.preco_entrada + 0.0005):
                    delta = preco_high - self.preco_entrada
                    novo_sl = (self.preco_entrada - self.sl_pips) + delta
                    
                    # SE ELE CONSEGUIU SUBIR A PROTEÇÃO, GANHA UM PONTINHO! (Reforço Positivo)
                    if novo_sl > self.alvo_sl:
                        recompensa += 0.0001  # Moedinha do Flippy Bird!
                        self.alvo_sl = novo_sl
                        self.alvo_tp = preco_high + self.tp_pips
                
                # Checa se Bateu no Stop ou Take
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
                # Trailing Dinâmico Proporcional (Só desce se der um respiro de 5 pips)
                if preco_low < (self.preco_entrada - 0.0005):
                    delta = self.preco_entrada - preco_low
                    novo_sl = (self.preco_entrada + self.sl_pips) - delta
                    
                    # SE ELE CONSEGUIU DESCER A PROTEÇÃO, GANHA UM PONTINHO!
                    if novo_sl < self.alvo_sl:
                        recompensa += 0.0001  # Moedinha do Flippy Bird!
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
        # 2. INTERPRETAÇÃO DA AÇÃO DO AGENTE (Se não estiver com trade fechando agora)
        # =====================================================================
        if self.posicao_aberta is None and not fechou_trade_neste_passo:
            
            # REGRA DA FASE: Se estourou o limite de mortes na hora, proíbe novas entradas (Força Hold)
            if self.mortes_na_hora >= self.limite_mortes_hora:
                action = 1  # Força Hold porque perdeu as vidas da fase!
            
            if action == 0:  # Tentar abrir VENDA
                self.posicao_aberta = "VENDA"
                self.preco_entrada = row_atual["close"]
                self.alvo_sl = self.preco_entrada + self.sl_pips
                self.alvo_tp = self.preco_entrada - self.tp_pips
                self.duracao_trade = 0
                recompensa -= (self.custo_transacao / self.preco_entrada)
                
            elif action == 2:  # Tentar abrir COMPRA
                self.posicao_aberta = "COMPRA"
                self.preco_entrada = row_atual["close"]
                self.alvo_sl = self.preco_entrada - self.sl_pips
                self.alvo_tp = self.preco_entrada + self.tp_pips
                self.duracao_trade = 0
                recompensa -= (self.custo_transacao / self.preco_entrada)
            
            # Se action == 1 (Hold), ele apenas plana sem abrir nada.

        # Avança para o próximo candle de M5
        self.passo_atual += 1
        finalizado = self.passo_atual >= len(self.df) - 1
        nova_obs = self._pegar_observacao() if not finalizado else np.zeros((self.num_features,), dtype=np.float32)
        
        return nova_obs, recompensa, finalizado, False, {}