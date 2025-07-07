
import argparse
import os
import csv
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def train_regression_model(n_samples, random_state, output_path, log_file):
    print(f"--- 回帰モデルの学習を開始します (n_samples: {n_samples}, random_state: {random_state}) ---")

    # ダミーデータの生成
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * 2 # y = 2x + 1 + ノイズ

    # データを訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # モデルの定義と学習
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("--- 学習が完了しました ---")

    # モデルの評価
    print("--- 学習済みモデルの評価 ---")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # ログの記録
    if log_file:
        print(f"--- 実験結果を記録中: {log_file} ---")
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'n_samples', 'random_state', 'mean_squared_error', 'output_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': n_samples,
                'random_state': random_state,
                'mean_squared_error': f"{mse:.4f}",
                'output_path': output_path
            })
        print("--- 記録が完了しました ---")

    # output_pathは受け取るが、このスクリプトではモデルの保存は行わない
    if output_path:
        print(f"注意: output_pathが指定されましたが、このスクリプトはモデルを保存しません。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='シンプルな回帰モデルの学習スクリプト')
    parser.add_argument('--n_samples', type=int, default=100, help='生成するサンプル数')
    parser.add_argument('--random_state', type=int, default=42, help='乱数シード')
    parser.add_argument('--output_path', type=str, default=None, help='学習済みモデルの保存先パス (このスクリプトでは使用されません)')
    parser.add_argument('--log_file', type=str, default=None, help='実験結果の記録用CSVファイル')
    args = parser.parse_args()

    train_regression_model(n_samples=args.n_samples, random_state=args.random_state, output_path=args.output_path, log_file=args.log_file)
