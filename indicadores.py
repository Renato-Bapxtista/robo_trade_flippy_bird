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
    """Calcula M5 e injeta tendências macro (H1 e D1) usando Shift para evitar Lookahead."""
    obrigatorias = {"time", "open", "high", "low", "close", "tick_volume"}
    faltantes = obrigatorias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Dados sem colunas obrigatórias: {sorted(faltantes)}")
        
    dados = df.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(dados['time']):
        dados['time'] = pd.to_datetime(dados['time'])
    
    # =========================================================================
    # 🕒 1. CÁLCULO MACRO H1
    # =========================================================================
    df_h1 = dados.set_index("time").resample("1h", closed='left', label='left').agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    
    df_h1["ema_21_h1"] = df_h1["close"].ewm(span=21, adjust=False).mean()
    df_h1["ema_200_h1"] = df_h1["close"].ewm(span=200, adjust=False).mean()
    df_h1["rsi_14_h1"] = calcular_rsi(df_h1["close"])
    
    df_h1["feat_macro_cruzamento"] = (df_h1["ema_21_h1"] - df_h1["ema_200_h1"]) / df_h1["ema_200_h1"]
    df_h1["feat_macro_rsi"] = (df_h1["rsi_14_h1"] - 50.0) / 50.0
    
    df_h1_features = df_h1[["time", "feat_macro_cruzamento", "feat_macro_rsi"]].copy()
    # SHIFT(1): Desloca o H1 para a próxima hora. O M5 de 10h vai ler os dados que fecharam 09h.
    df_h1_features.iloc[:, 1:] = df_h1_features.iloc[:, 1:].shift(1)
    df_h1_features = df_h1_features.dropna().rename(columns={"time": "hora_chave"})

    # =========================================================================
    # 📅 2. CÁLCULO MACRO DIÁRIO (D1)
    # =========================================================================
    df_d1 = dados.set_index("time").resample("1D", closed='left', label='left').agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    
    df_d1["ema_21_d1"] = df_d1["close"].ewm(span=21, adjust=False).mean()
    df_d1["ema_200_d1"] = df_d1["close"].ewm(span=200, adjust=False).mean()
    df_d1["rsi_14_d1"] = calcular_rsi(df_d1["close"])
    
    df_d1["feat_diario_cruzamento"] = (df_d1["ema_21_d1"] - df_d1["ema_200_d1"]) / df_d1["ema_200_d1"]
    df_d1["feat_diario_rsi"] = (df_d1["rsi_14_d1"] - 50.0) / 50.0
    
    df_d1_features = df_d1[["time", "feat_diario_cruzamento", "feat_diario_rsi"]].copy()
    # SHIFT(1): Desloca o D1 para o próximo dia útil. O M5 de hoje só vê como o mercado fechou ontem.
    df_d1_features.iloc[:, 1:] = df_d1_features.iloc[:, 1:].shift(1)
    df_d1_features = df_d1_features.dropna().rename(columns={"time": "dia_chave"})

    # =========================================================================
    # ⚡ 3. CÁLCULOS DO CALOR DO MOMENTO (M5)
    # =========================================================================
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
    
    # Engenharia de Features M5
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
    # 🧩 4. SINCRONISMO DOS TEMPOS (MERGE H1 e D1 NO M5)
    # =========================================================================
    dados["hora_chave"] = dados["time"].dt.floor("1h")
    dados["dia_chave"] = dados["time"].dt.floor("1D")
    
    # Anexando H1
    dados = pd.merge(dados, df_h1_features, on="hora_chave", how="left")
    # Anexando D1
    dados = pd.merge(dados, df_d1_features, on="dia_chave", how="left")
    
    # Forward Fill nas novas colunas para garantir cobertura em caso de feriados/buracos
    colunas_macro = ["feat_macro_cruzamento", "feat_macro_rsi", "feat_diario_cruzamento", "feat_diario_rsi"]
    dados[colunas_macro] = dados[colunas_macro].ffill().fillna(0.0)
    
    # Limpeza final
    dados = dados.drop(columns=["hora_chave", "dia_chave"])
    
    return dados.dropna().reset_index(drop=True)