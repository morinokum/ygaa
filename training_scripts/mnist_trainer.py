
import tensorflow as tf
import argparse
import os
import csv
from datetime import datetime
import numpy as np

def train_mnist(epochs, batch_size, output_path, log_file, input_data_path):
    # 1. データのロードと前処理
    if input_data_path:
        print(f"--- データロード中: {input_data_path} ---")
        data = np.load(input_data_path)
        x_train = data['X_train']
        y_train = data['y_train']
        x_test = data['X_test']
        y_test = data['y_test']
    else:
        print("--- MNISTデータセットをロード中 ---")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 画像データを0-1の範囲に正規化
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # モデルが扱いやすいように画像の次元を追加 (もし必要なら)
    if len(x_train.shape) == 3: # (samples, height, width) の場合
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

    # ラベルをカテゴリカル形式に変換 (例: 5 -> [0,0,0,0,0,1,0,0,0,0])
    # y_trainが既にone-hotエンコーディングされているか確認
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # 2. モデルの定義
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 3. モデルのコンパイル
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. モデルの学習
    print(f"--- MNISTモデルの学習を開始します (epochs: {epochs}, batch_size: {batch_size}) ---")
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=2 # 学習の進捗表示を少し簡潔に
    )
    print("--- 学習が完了しました ---")

    # 5. モデルの評価
    print("--- 学習済みモデルの評価 ---")
    score = model.evaluate(x_test, y_test, verbose=0)
    test_loss = score[0]
    test_accuracy = score[1]
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # 6. モデルの保存
    if output_path:
        print(f"--- 学習済みモデルを保存中: {output_path} ---")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save(output_path)
        print("--- 保存が完了しました ---")

    # 7. 結果のロギング
    if log_file:
        print(f"--- 実験結果を記録中: {log_file} ---")
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'epochs', 'batch_size', 'test_loss', 'test_accuracy', 'output_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'epochs': epochs,
                'batch_size': batch_size,
                'test_loss': f"{test_loss:.4f}",
                'test_accuracy': f"{test_accuracy:.4f}",
                'output_path': output_path
            })
        print("--- 記録が完了しました ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNISTモデルの学習スクリプト')
    parser.add_argument('--epochs', type=int, default=5, help='学習のエポック数')
    parser.add_argument('--batch_size', type=int, default=32, help='学習のバッチサイズ')
    parser.add_argument('--output_path', type=str, default=None, help='学習済みモデルの保存先パス')
    parser.add_argument('--log_file', type=str, default=None, help='実験結果の記録用CSVファイル')
    parser.add_argument('--input_data_path', type=str, default=None, help='入力データファイルへのパス (NPZ形式)')
    args = parser.parse_args()

    train_mnist(epochs=args.epochs, batch_size=args.batch_size, output_path=args.output_path, log_file=args.log_file, input_data_path=args.input_data_path)
