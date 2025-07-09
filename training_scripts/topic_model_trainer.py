
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
import os

# プロジェクトのルートディレクトリを基準にパスを設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'labeled_user_logs.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'topic_classifier.joblib')

def train_topic_model():
    """
    ユーザーの対話ログからトピック分類モデルを学習し、保存する。
    """
    print("モデルの学習を開始します...")

    # 1. データの読み込み
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"学習データを読み込みました: {DATA_FILE_PATH}")
    except FileNotFoundError:
        print(f"エラー: 学習データが見つかりません: {DATA_FILE_PATH}")
        return

    # 簡単な前処理
    df = df.dropna(subset=['text', 'topic'])
    if df.empty:
        print("エラー: 学習データが空です。")
        return

    X = df['text']
    y = df['topic']

    print(f"学習サンプル数: {len(df)}")
    print("トピックの内訳:")
    print(y.value_counts())

    # 2. モデルの構築と学習
    # TfidfVectorizer: テキストを数値ベクトルに変換
    # LogisticRegression: 分類モデル
    print("TF-IDFベクトル化とロジスティック回帰モデルを構築します。")
    pipeline = make_pipeline(
        TfidfVectorizer(min_df=1), # 低頻度すぎる単語は無視しない
        LogisticRegression(random_state=42)
    )

    print("モデルの学習中...")
    pipeline.fit(X, y)
    print("モデルの学習が完了しました。")

    # 3. モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"学習済みモデルを保存しました: {MODEL_PATH}")

if __name__ == '__main__':
    train_topic_model()
