
import argparse
import numpy as np
import os

def preprocess_data(output_data_path):
    print(f"--- データ前処理を開始します。出力パス: {output_data_path} ---")

    # ダミーの訓練データとテストデータを生成
    # 実際のシナリオでは、ここで実際のデータロード、クレンジング、特徴量エンジニアリングなどが行われます。
    num_samples_train = 1000
    num_samples_test = 200
    num_features = 10

    X_train = np.random.rand(num_samples_train, num_features).astype(np.float32)
    y_train_int = np.random.randint(0, 10, num_samples_train) # 0-9の整数ラベルを生成
    y_train = np.eye(10)[y_train_int].astype(np.float32) # One-hotエンコーディング

    X_test = np.random.rand(num_samples_test, num_features).astype(np.float32)
    y_test_int = np.random.randint(0, 10, num_samples_test) # 0-9の整数ラベルを生成
    y_test = np.eye(10)[y_test_int].astype(np.float32) # One-hotエンコーディング

    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_data_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # データをNPZ形式で保存
    np.savez(output_data_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("--- データ前処理が完了し、データが保存されました ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ダミーデータを生成し保存するスクリプト')
    parser.add_argument('--output_data_path', type=str, required=True, help='処理済みデータの保存先パス (例: data/processed_data.npz)')
    parser.add_argument('--log_file', type=str, default=None, help='実験結果の記録用CSVファイル (このスクリプトでは使用されません)')
    args = parser.parse_args()

    preprocess_data(output_data_path=args.output_data_path)
