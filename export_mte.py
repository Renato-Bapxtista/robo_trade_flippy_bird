from dados import obter_dados_mt5
import pandas as pd
from datetime import datetime, timezone
import os

# Ajuste `ativo`, `inicio` e `fim` conforme necessário.
# Por padrão a função usa timeframe M5 e início em 2024-01-01.
df = obter_dados_mt5(ativo='EURUSD', inicio=datetime(2024, 1, 1, tzinfo=timezone.utc))
out_dir = 'para'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'm5.csv')
df.to_csv(out_path, index=False)
print(f'salvo em {out_path}')