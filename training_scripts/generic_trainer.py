import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras

def train_model(dataset_path, output_path, log_file, epochs, batch_size, learning_rate, optimizer_type):
    print(f"Training model with dataset: {dataset_path}")
    print(f"Output model path: {output_path}")
    print(f"Log file: {log_file}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Optimizer: {optimizer_type}")

    try:
        # dataset_pathからデータを読み込む
        # ここではCSVファイルを想定。必要に応じて他の形式にも対応
        df = pd.read_csv(dataset_path, header=None, skiprows=1) # 最初の行をスキップ

        # 最後の列がターゲット変数、それ以外が特徴量と仮定
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # ターゲット変数が文字列の場合、数値にエンコード
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print("Target variable encoded to numerical values.")

        # 特徴量Xに文字列が残っていないか確認し、あればエラーを出すか、適切に処理する
        for col in X.columns:
            if X[col].dtype == 'object':
                raise ValueError(f"Feature column '{col}' contains non-numeric data after target separation.")

        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Kerasモデルの構築
        num_classes = len(set(y_train))
        model = keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

        # オプティマイザの選択
        if optimizer_type.lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate) # デフォルトはSGD

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # モデルの訓練
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # モデルを保存
        model.save(output_path)
        print(f"Model trained and saved to {output_path}")

        # 訓練データでの精度をログに記録
        _, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        print(f"Training Accuracy: {train_accuracy}")

        # テストデータでの評価
        print("\n--- Evaluating model on test data ---")
        y_pred = model.predict(X_test)
        y_pred_classes = tf.argmax(y_pred, axis=1)

        test_accuracy = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes)

        print(f"Test Accuracy: {test_accuracy}")
        print("Classification Report:\n", report)

        with open(log_file, 'a') as f:
            f.write(f"Training completed for {output_path}. Training Accuracy: {train_accuracy}\n")
            f.write(f"Evaluation completed for {output_path}. Test Accuracy: {test_accuracy}\n")
            f.write(f"Classification Report:\n{report}\n")

    except Exception as e:
        print(f"Error during training and evaluation: {e}")
        # エラーログをファイルに書き込むことも可能

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generic Model Trainer Script.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the log file.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--optimizer_type", type=str, default="adam", help="Optimizer type.")

    args = parser.parse_args()

    train_model(args.dataset_path, args.output_path, args.log_file, args.epochs, args.batch_size, args.learning_rate, args.optimizer_type)
