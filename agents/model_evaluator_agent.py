import os
import argparse
import numpy as np
import tensorflow as tf
import csv
from datetime import datetime
import sys

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# デフォルト設定
DEFAULT_CONFIG = {
    "model_path": os.path.join(PROJECT_ROOT, "trained_models", "mnist_model_latest.keras"),
    "test_data_path": None, # 評価に使用するテストデータのパス
    "evaluation_log_file": os.path.join(PROJECT_ROOT, "logs", "model_evaluation_log.csv")
}

def main(args, config):
    """
    学習済みモデルをロードし、テストデータで評価するエージェント。
    """
    print("Model Evaluator Agent: 開始")

    model_path = config.get("model_path", DEFAULT_CONFIG["model_path"])
    test_data_path = config.get("test_data_path", DEFAULT_CONFIG["test_data_path"])
    evaluation_log_file = config.get("evaluation_log_file", DEFAULT_CONFIG["evaluation_log_file"])

    print(f"Debug: model_path (absolute) = {model_path}")
    print(f"Debug: os.path.exists(model_path) = {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        print(f"エラー: 学習済みモデルが見つかりません: {model_path}", file=sys.stderr)
        return

    # 1. モデルのロード
    print(f"--- モデルをロード中: {model_path} ---")
    try:
        model = tf.keras.models.load_model(model_path)
        print("--- モデルのロードが完了しました ---")
    except Exception as e:
        print(f"エラー: モデルのロードに失敗しました: {e}", file=sys.stderr)
        return

    # 2. テストデータのロードと前処理
    print(f"--- テストデータをロード中: {test_data_path} ---")
    try:
        # MNISTデータセットを想定
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # 画像データを0-1の範囲に正規化
        x_test = x_test.astype("float32") / 255

        # モデルが扱いやすいように画像の次元を追加 (もし必要なら)
        if len(x_test.shape) == 3: # (samples, height, width) の場合
            x_test = x_test[..., tf.newaxis]

        # ラベルをカテゴリカル形式に変換
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        print("--- テストデータのロードと前処理が完了しました ---")
    except Exception as e:
        print(f"エラー: テストデータのロードまたは前処理に失敗しました: {e}", file=sys.stderr)
        return

    # 3. モデルの評価
    print("--- モデルを評価中 ---")
    
    # 強化学習モデルの場合は評価をスキップ（別途シミュレーションで評価）
    if "reinforce" in model_path.lower(): # モデルパスに"reinforce"が含まれるかで簡易的に判別
        print("強化学習モデルのため、評価をスキップします。")
        loss = "N/A"
        accuracy = "N/A"
    else:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"評価結果 - 損失: {loss:.4f}, 精度: {accuracy:.4f}")
    print("--- モデル評価が完了しました ---")

    # 4. 評価結果のロギング
    if evaluation_log_file:
        print(f"--- 評価結果を記録中: {evaluation_log_file} ---")
        log_dir = os.path.dirname(evaluation_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_exists = os.path.isfile(evaluation_log_file)
        with open(evaluation_log_file, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'model_path', 'test_data_path', 'loss', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': model_path,
                'test_data_path': test_data_path,
                'loss': f"{loss:.4f}" if isinstance(loss, float) else loss,
                'accuracy': f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy
            })
        print("--- 記録が完了しました ---")

    print("Model Evaluator Agent: 終了")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='学習済みモデル評価エージェント')
    parser.add_argument('--model_path', type=str, default=DEFAULT_CONFIG["model_path"], help='評価する学習済みモデルのパス')
    parser.add_argument('--test_data_path', type=str, default=DEFAULT_CONFIG["test_data_path"], help='評価に使用するテストデータのパス')
    parser.add_argument('--evaluation_log_file', type=str, default=DEFAULT_CONFIG["evaluation_log_file"], help='評価結果の記録用CSVファイル')
    args = parser.parse_args()
    
    config = {
        "model_path": args.model_path,
        "test_data_path": args.test_data_path,
        "evaluation_log_file": args.evaluation_log_file
    }
    main([], config)