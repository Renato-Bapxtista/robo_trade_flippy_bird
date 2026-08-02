from datetime import datetime, timezone
import time
import os
import MetaTrader5 as mt5
import pandas as pd

def _utc(data):
    if data.tzinfo is None:
        return data.replace(tzinfo=timezone.utc)
    return data.astimezone(timezone.utc)


def obter_dados_mt5(ativo="EURUSD", timeframe=mt5.TIMEFRAME_M5, inicio=datetime(2024, 1, 1, tzinfo=timezone.utc), fim=None, retries: int = 6, wait_seconds: int = 3, local_fallback: str | None = None) -> pd.DataFrame:
    """Tenta obter candles do MT5 com várias tentativas.

    Se não conseguir, e `local_fallback` for um caminho válido, carrega CSV local.
    """
    if not mt5.initialize():
        raise RuntimeError(f"Não foi possível inicializar MT5: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(ativo, True):
            raise RuntimeError(f"Ativo indisponível no MT5: {ativo}")

        # força o MT5 a sincronizar histórico abrindo o ativo no Market Watch
        mt5.market_book_add(ativo)

        inicio_utc = _utc(inicio)
        fim_utc = _utc(fim or datetime.now(timezone.utc))

        if inicio_utc >= fim_utc:
            raise ValueError("A data inicial deve ser anterior à data final.")

        print(f"Sincronizando histórico de {ativo} desde {inicio_utc.year}... Aguarde.")

        dados = None
        for attempt in range(retries):
            dados = mt5.copy_rates_range(ativo, timeframe, inicio_utc, fim_utc)
            if dados is not None and len(dados) > 0:
                break
            wait = wait_seconds * (attempt + 1)
            print(f"Dados ainda sendo baixados pelo MT5... tentativa {attempt+1}/{retries}. Aguardando {wait}s.")
            time.sleep(wait)

        if dados is None or len(dados) == 0:
            # fallback para CSV local, se fornecido
            if local_fallback and os.path.exists(local_fallback):
                print(f"MT5 falhou — carregando fallback local: {local_fallback}")
                df = pd.read_csv(local_fallback)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                return df

            # informação adicional para o usuário sobre possíveis causas
            last_err = mt5.last_error()
            raise RuntimeError(
                "MT5 não retornou candles. Verifique: terminal logado, símbolo no Market Watch, 'Max Bars in History' nas opções do MT5. "
                f"Último erro MT5: {last_err}"
            )

        df = pd.DataFrame(dados)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        df["price_range"] = df["high"] - df["low"]
        df["price_volume"] = df["price_range"] * df["tick_volume"]
        df["real_volume"] = df["real_volume"]

        return df
    finally:
        try:
            mt5.market_book_release(ativo)
        except Exception:
            pass
        mt5.shutdown()


def separar_dados_temporal(dados: pd.DataFrame, proporcao_treino: float = 0.70, proporcao_validacao: float = 0.15, minimo_por_conjunto: int = 250) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < proporcao_treino < 1 or not 0 < proporcao_validacao < 1:
        raise ValueError("Proporções inválidas.")
    if proporcao_treino + proporcao_validacao >= 1:
        raise ValueError("Treino e validação devem deixar uma parcela para teste.")
    if "time" not in dados.columns:
        raise ValueError("Dados sem a coluna obrigatória 'time'.")
        
    dados_ordenados = dados.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    
    fim_treino = int(len(dados_ordenados) * proporcao_treino)
    fim_validacao = fim_treino + int(len(dados_ordenados) * proporcao_validacao)
    
    treino = dados_ordenados.iloc[:fim_treino].copy()
    validacao = dados_ordenados.iloc[fim_treino:fim_validacao].copy()
    teste = dados_ordenados.iloc[fim_validacao:].copy()
    
    if min(len(treino), len(validacao), len(teste)) < minimo_por_conjunto:
        raise ValueError("Volume de dados insuficiente. Puxe um período maior no MT5.")
        
    return treino, validacao, teste

# --- BLOCO DE TESTE ---
# Este bloco roda apenas quando você executa este arquivo direto
if __name__ == "__main__":
    print("Testando a nossa Peça 1...")
    # Buscando dados de teste (usando EURUSD como padrão do seu código)
    df_completo = obter_dados_mt5(ativo="EURUSD", inicio=datetime(2024, 1, 1, tzinfo=timezone.utc))
    print(f"Total de linhas baixadas: {len(df_completo)}")
    
    df_treino, df_val, df_teste = separar_dados_temporal(df_completo)
    print(f"Linhas para Treino: {len(df_treino)}")
    print(f"Linhas para Validação: {len(df_val)}")
    print(f"Linhas para Teste: {len(df_teste)}")
