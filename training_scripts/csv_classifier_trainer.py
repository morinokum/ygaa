import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# utilsディレクトリをパスに追加
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
from csv_analyzer import analyze_csv_features

def train_csv_classifier(output_model_path):
    print("--- CSV分類モデルの学習を開始します ---")

    # 既知のログファイルのパスとラベル
    # 実際の運用では、より多くの多様なログファイルで学習させるべきです
    log_samples = [
        (os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log.csv"), "mnist"),
        (os.path.join(PROJECT_ROOT, "logs", "reinforce_cartpole_log.csv"), "reinforce"),
    ]

    X = [] # 特徴量
    y = [] # ラベル

    for file_path, label in log_samples:
        if os.path.exists(file_path):
            print(f"特徴量を抽出中: {file_path} ({label})")
            features = analyze_csv_features(file_path)
            # 特徴量の辞書をリストに変換（順序を保証するためキーをソート）
            feature_vector = [features[key] for key in sorted(features.keys())]
            X.append(feature_vector)
            y.append(label)
        else:
            print(f"警告: ファイルが見つかりません: {file_path}。スキップします。")

    if not X:
        print("エラー: 学習データがありません。モデルを学習できません。")
        return

    X = np.array(X)
    y = np.array(y)

    # データを訓練セットとテストセットに分割
    # サンプル数が少ないため、ここでは分割せず全データで学習
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X, y

    # ロジスティック回帰モデルの学習
    print("ロジスティック回帰モデルを学習中...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    print("学習完了。")

    # モデルの評価 (ここでは訓練データに対する精度)
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"訓練データに対する精度: {accuracy:.2f}")

    # 学習済みモデルの保存
    output_dir = os.path.dirname(output_model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model, output_model_path)
    print(f"学習済みモデルを保存しました: {output_model_path}")
    print("--- CSV分類モデルの学習が完了しました ---")

if __name__ == '__main__':
    # デフォルトの出力パス
    default_output_path = os.path.join(PROJECT_ROOT, "trained_models", "csv_classifier_model.joblib")
    train_csv_classifier(default_output_path)