import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from h1_train import build_h1
from indicadores import preparar_dados_mercado
from dados import separar_dados_temporal
from ambiente import AmbienteTrading
from stable_baselines3 import PPO


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def h1_diagnostics(m5_df, models_path, out_dir):
    ensure_dir(out_dir)
    df_h1 = build_h1(m5_df)

    # build targets
    df_h1['alvo_direcao'] = (df_h1['close'].shift(-1) > df_h1['open'].shift(-1)).astype(int)
    df_h1 = df_h1.dropna(subset=['alvo_direcao']).reset_index(drop=True)

    cols = ['feat_macro_cruzamento', 'feat_macro_rsi', 'open', 'high', 'low', 'close']
    X = df_h1[cols]
    y_true = df_h1['alvo_direcao']

    if not os.path.exists(models_path):
        print(f"H1 model not found at {models_path}")
        return

    clf, reg_tam, reg_dist = joblib.load(models_path)

    X_array = X.to_numpy()
    y_pred = clf.predict(X_array)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(os.path.join(out_dir, 'h1_classification_report.txt'), 'w') as f:
        f.write('=== Classification Report ===\n')
        f.write(report)
        f.write('\n\n=== Confusion Matrix ===\n')
        f.write(str(cm))

    # plot confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('H1 Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'h1_confusion_matrix.png'))
    plt.close()

    # feature importances
    try:
        importances = clf.feature_importances_
        fi = pd.Series(importances, index=cols).sort_values(ascending=False)
        plt.figure(figsize=(6,4))
        fi.plot(kind='bar')
        plt.title('H1 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'h1_feature_importances.png'))
        plt.close()
    except Exception:
        pass

    print('H1 diagnostics saved to', out_dir)


def trade_diagnostics(m5_df, ppo_path, out_dir):
    ensure_dir(out_dir)
    # compute features used by environment
    df_features = preparar_dados_mercado(m5_df)
    # split temporally
    _, _, df_test = separar_dados_temporal(df_features)

    env = AmbienteTrading(df_test)
    model = PPO.load(ppo_path)

    obs, _ = env.reset()
    finalizado = False

    rewards = []
    durations = []
    action_counts = {0:0,1:0,2:0}

    current_trade_duration = 0
    while not finalizado:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1

        new_obs, reward, finalizado, trunc, info = env.step(action)

        # record only closed trades (non-zero reward)
        if action in (0,2) and reward != 0:
            rewards.append(float(reward))
        # track durations via env internals if present
        current_trade_duration += 1
        if reward != 0:
            durations.append(current_trade_duration)
            current_trade_duration = 0

        obs = new_obs

    # stats and plots
    if len(rewards) > 0:
        s = pd.Series(rewards)
        with open(os.path.join(out_dir, 'trade_reward_stats.txt'), 'w') as f:
            f.write(s.describe().to_string())

        plt.figure()
        sns.histplot(s, bins=50, kde=False)
        plt.title('Trade Reward Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'trade_rewards_hist.png'))
        plt.close()

    if len(durations) > 0:
        sd = pd.Series(durations)
        with open(os.path.join(out_dir, 'trade_duration_stats.txt'), 'w') as f:
            f.write(sd.describe().to_string())

        plt.figure()
        sns.histplot(sd, bins=50)
        plt.title('Trade Duration Distribution (candles)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'trade_durations_hist.png'))
        plt.close()

    with open(os.path.join(out_dir, 'action_counts.txt'), 'w') as f:
        f.write(str(action_counts))

    print('Trade diagnostics saved to', out_dir)


def main():
    p = argparse.ArgumentParser(description='Diagnostics for H1 models and PPO trade behavior')
    p.add_argument('--m5-csv', default='para/m5.csv', help='Path to M5 CSV to analyze (default: para/m5.csv)')
    p.add_argument('--h1-model', default='models/h1_models.joblib')
    p.add_argument('--ppo-model', default=None, help='Path to PPO model zip (optional)')
    p.add_argument('--out-dir', default='diagnostics')
    args = p.parse_args()

    if not os.path.exists(args.m5_csv):
        raise FileNotFoundError(
            f"Arquivo M5 não encontrado: {args.m5_csv}.\n" \
            "Crie-o com export_mte.py ou informe --m5-csv para o caminho correto."
        )

    m5_df = pd.read_csv(args.m5_csv)
    if 'time' in m5_df.columns:
        m5_df['time'] = pd.to_datetime(m5_df['time'])

    h1_diagnostics(m5_df, args.h1_model, args.out_dir)
    if args.ppo_model:
        trade_diagnostics(m5_df, args.ppo_model, args.out_dir)


if __name__ == '__main__':
    main()
