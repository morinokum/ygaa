import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path, dataset_path, log_file):
    print(f"Evaluating model: {model_path}")
    print(f"Using dataset: {dataset_path}")
    print(f"Log file: {log_file}")

    try:
        # モデルをロード
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")

        # データセットをロード (訓練時と同じ方法で)
        df = pd.read_csv(dataset_path, header=None, skiprows=1) # 最初の行をスキップ

        # 最後の列がターゲット変数、それ以外が特徴量と仮定
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # ターゲット変数が文字列の場合、数値にエンコード (訓練時と同じエンコーダを使用すべきだが、ここでは簡略化)
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        # 訓練データとテストデータに分割 (訓練時と同じrandom_stateを使用)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 予測
        y_pred = model.predict(X_test)

        # 評価指標を計算
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Test Accuracy: {accuracy}")
        print("Classification Report:\n", report)

        with open(log_file, 'a') as f:
            f.write(f"\nEvaluation completed for {model_path}. Test Accuracy: {accuracy}\n")
            f.write(f"Classification Report:\n{report}\n")

    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generic Model Evaluator Script.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset for evaluation.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the log file.")

    args = parser.parse_args()

    evaluate_model(args.model_path, args.dataset_path, args.log_file)
