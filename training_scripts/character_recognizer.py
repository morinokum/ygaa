
import tensorflow as tf
import argparse
import os
import csv
from datetime import datetime
import numpy as np
import json

def train_character_recognizer(epochs, batch_size, output_path, log_file, learning_rate=0.001, optimizer_type='adam'):
    # 1. データのロードと前処理
    print(f"--- 文字画像データセットをロード中: data/neo_world_characters.npz ---")
    data = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "neo_world_characters.npz"))
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']

    # 画像データを0-1の範囲に正規化
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # 画像の形状を (samples, height, width, channels) に変更
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # ユニークな文字の数を取得
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    char_to_label_path = os.path.join(project_root, "data", "character_images", "char_to_label.json")
    with open(char_to_label_path, "r", encoding="utf-8") as f:
        char_to_label = json.load(f)
    num_classes = len(char_to_label)

    # ラベルをカテゴリカル形式に変換
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    # 2. モデルの定義
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax') # 出力層のユニット数を変更
    ])

    # 3. モデルのコンパイル
    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        print(f"警告: 未知のオプティマイザタイプ '{optimizer_type}' です。Adamを使用します。", file=sys.stderr)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. モデルの学習
    print(f"--- 文字認識モデルの学習を開始します (epochs: {epochs}, batch_size: {batch_size}) ---")
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
    parser = argparse.ArgumentParser(description='文字認識モデルの学習スクリプト')
    parser.add_argument('--epochs', type=int, default=5, help='学習のエポック数')
    parser.add_argument('--batch_size', type=int, default=32, help='学習のバッチサイズ')
    parser.add_argument('--output_path', type=str, default=None, help='学習済みモデルの保存先パス')
    parser.add_argument('--log_file', type=str, default=None, help='実験結果の記録用CSVファイル')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学習率')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='オプティマイザのタイプ (adam, sgd)')
    args = parser.parse_args()

    train_character_recognizer(epochs=args.epochs, batch_size=args.batch_size, output_path=args.output_path, log_file=args.log_file, learning_rate=args.learning_rate, optimizer_type=args.optimizer_type)
