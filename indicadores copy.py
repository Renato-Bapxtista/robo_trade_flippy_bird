import numpy as np
import pandas as pd

def calcular_rsi(close: pd.Series, periodo: int = 14) -> pd.Series:
    """RSI de Wilder calculado somente com barras já encerradas."""
    variacao = close.diff()
    ganhos = variacao.clip(lower=0)
    perdas = -variacao.clip(upper=0)
    
    ganho_medio = ganhos.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    perda_media = perdas.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    
    rs = ganho_medio / perda_media.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.mask((perda_media == 0) & (ganho_medio > 0), 100.0)
    rsi = rsi.mask((ganho_medio == 0) & (perda_media > 0), 0.0)
    return rsi.fillna(50.0)

def preparar_dados_mercado(df: pd.DataFrame, janela_sup_res: int = 50) -> pd.DataFrame:
    """Calcula indicadores M5 e injeta a tendência macro do gráfico H1 sem lookahead bias."""
    obrigatorias = {"time", "open", "high", "low", "close", "tick_volume"}
    faltantes = obrigatorias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Dados sem colunas obrigatórias: {sorted(faltantes)}")
        
    # Garante que 'time' seja datetime para evitar erros no .dt
    dados = df.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(dados['time']):
        dados['time'] = pd.to_datetime(dados['time'])
    
    # =========================================================================
    # 🕒 CÁLCULO MULTITIMEFRAME (H1 a partir do M5)
    # =========================================================================
    df_h1 = dados.set_index("time").resample("1h", closed='left', label='left').agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()
    
    # Indicadores do Gráfico de 1 Hora (Macro)
    df_h1["ema_21_h1"] = df_h1["close"].ewm(span=21, adjust=False).mean()
    df_h1["ema_200_h1"] = df_h1["close"].ewm(span=200, adjust=False).mean()
    df_h1["rsi_14_h1"] = calcular_rsi(df_h1["close"])
    
    # Features Relativas do H1
    df_h1["feat_macro_cruzamento"] = (df_h1["ema_21_h1"] - df_h1["ema_200_h1"]) / df_h1["ema_200_h1"]
    df_h1["feat_macro_rsi"] = (df_h1["rsi_14_h1"] - 50.0) / 50.0
    
    # Guardamos apenas o tempo e as features do H1
    df_h1_features = df_h1[["time", "feat_macro_cruzamento", "feat_macro_rsi"]].copy()
    
    # [CORREÇÃO CRÍTICA]: Deslocamos o tempo H1 em 1 hora para frente.
    # O candle de 09:00 (que fecha às 10:00) será disponibilizado apenas a partir das 10:00.
    df_h1_features["hora_chave"] = df_h1_features["time"] + pd.Timedelta(hours=1)
    df_h1_features = df_h1_features.drop(columns=["time"])
    # =========================================================================
    
    # --- 1. Cálculos de Indicadores M5 (Calor do Momento) ---
    dados["atr_14"] = (dados["high"] - dados["low"]).rolling(14, min_periods=14).mean()
    dados["media_atr"] = dados["atr_14"].rolling(20, min_periods=20).mean()
    dados["ema_21"] = dados["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    dados["ema_200"] = dados["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    dados["rsi_14"] = calcular_rsi(dados["close"])
    dados["suporte"] = dados["low"].rolling(janela_sup_res, min_periods=janela_sup_res).min()
    dados["resistencia"] = dados["high"].rolling(janela_sup_res, min_periods=janela_sup_res).max()
    dados["media_volume_20"] = dados["tick_volume"].rolling(20, min_periods=20).mean()
    dados["retorno_5"] = dados["close"].pct_change(periods=5)
    
    dados = dados.dropna().reset_index(drop=True)
    
    # --- 2. Engenharia de Features M5 ---
    dados["feat_dist_ema21"] = (dados["close"] - dados["ema_21"]) / dados["ema_21"]
    dados["feat_dist_ema200"] = (dados["close"] - dados["ema_200"]) / dados["ema_200"]
    dados["feat_dist_suporte"] = (dados["close"] - dados["suporte"]) / dados["suporte"]
    dados["feat_dist_resistencia"] = (dados["resistencia"] - dados["close"]) / dados["close"]
    dados["feat_rsi_normalizado"] = (dados["rsi_14"] - 50.0) / 50.0
    dados["feat_volume_relativo"] = dados["tick_volume"] / dados["media_volume_20"].replace(0, 1)
    dados["feat_volatilidade_relativa"] = dados["atr_14"] / dados["media_atr"].replace(0, 1)
    dados["feat_cruzamento_medias"] = (dados["ema_21"] - dados["ema_200"]) / dados["ema_200"]
    dados["feat_momento_5"] = dados["retorno_5"] * 100.0
    
    # =========================================================================
    # 🧩 SINCRONISMO DOS TEMPOS SEM LOOKAHEAD BIAS
    # =========================================================================
    dados["hora_chave"] = dados["time"].dt.floor("1h")
    
    # [CORREÇÃO]: Sintaxe do merge arrumada
    dados = pd.merge(dados, df_h1_features, on="hora_chave", how="left")
    
    # Preenche possíveis lacunas e remove as colunas de controle
    dados["feat_macro_cruzamento"] = dados["feat_macro_cruzamento"].ffill().fillna(0.0)
    dados["feat_macro_rsi"] = dados["feat_macro_rsi"].ffill().fillna(0.0)
    
    dados = dados.drop(columns=["hora_chave"])
    # =========================================================================
    
    return dados.dropna().reset_index(drop=True)

if __name__ == "__main__":
    from dados import obter_dados_mt5
    from datetime import datetime, timezone
    df_bruto = obter_dados_mt5(ativo="EURUSD", inicio=datetime(2024, 1, 1, tzinfo=timezone.utc))
    df_processado = preparar_dados_mercado(df_bruto)
    colunas_ia = [col for col in df_processado.columns if col.startswith("feat_")]
    print(f"Sucesso! Total de features (M5 + Macro H1): {len(colunas_ia)}")
    print("Novas colunas macro anexadas:", [c for c in colunas_ia if "macro" in c])
