
import argparse
import tensorflow as tf
import numpy as np
import os
import csv
from datetime import datetime
import sys
import json

def evaluate_model(model_path, input_data_path, log_file):
    print(f"--- モデル評価を開始します。モデル: {model_path}, データ: {input_data_path} ---")

    if not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません: {model_path}", file=sys.stderr)
        return

    # モデルのロード
    model = tf.keras.models.load_model(model_path)

    # データのロード
    print(f"--- 評価データロード中: {input_data_path} ---")
    data = np.load(input_data_path)
    x_test = data['X_test']
    y_test = data['y_test']

    # 画像データを0-1の範囲に正規化
    x_test = x_test.astype("float32") / 255

    # 画像の形状を (samples, height, width, channels) に変更
    x_test = x_test.reshape(-1, 28, 28, 1)

    # ユニークな文字の数を取得
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    char_to_label_path = os.path.join(project_root, "data", "character_images", "char_to_label.json")
    with open(char_to_label_path, "r", encoding="utf-8") as f:
        char_to_label = json.load(f)
    num_classes = len(char_to_label)

    # ラベルをカテゴリカル形式に変換
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    # モデルの評価
    score = model.evaluate(x_test, y_test, verbose=0)
    test_loss = score[0]
    test_accuracy = score[1]

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("--- モデル評価が完了しました ---")

    # 結果のロギング
    if log_file:
        print(f"--- 評価結果を記録中: {log_file} ---")
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'model_path', 'input_data_path', 'evaluated_loss', 'evaluated_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': model_path,
                'input_data_path': input_data_path,
                'evaluated_loss': f"{test_loss:.4f}",
                'evaluated_accuracy': f"{test_accuracy:.4f}"
            })
        print("--- 記録が完了しました ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='学習済みモデルを評価するスクリプト')
    parser.add_argument('--model_path', type=str, required=True, help='評価する学習済みモデルのパス')
    parser.add_argument('--input_data_path', type=str, default=None, help='評価用データファイルへのパス (NPZ形式)')
    parser.add_argument('--log_file', type=str, default=None, help='評価結果の記録用CSVファイル')
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path, input_data_path=args.input_data_path, log_file=args.log_file)
