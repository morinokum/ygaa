import os
import sys
import joblib
import numpy as np

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# utilsディレクトリをパスに追加
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
from csv_analyzer import analyze_csv_features

# デフォルト設定
DEFAULT_CONFIG = {
    "dataset_path": None, # 分析したいデータセットのCSVファイル
    "classifier_model_path": os.path.join(PROJECT_ROOT, "trained_models", "csv_classifier_model.joblib")
}

def main(args, config):
    """
    データセットのCSVファイルを分析し、適切なモデルタイプを推奨するエージェント。
    """
    print("Dataset Recommender Agent: 開始")

    dataset_path = config.get("dataset_path", DEFAULT_CONFIG["dataset_path"])
    classifier_model_path = config.get("classifier_model_path", DEFAULT_CONFIG["classifier_model_path"])

    if not dataset_path:
        print("エラー: 分析したいデータセットファイルが指定されていません。--agent-set dataset_path=<path_to_csv> で指定してください。", file=sys.stderr)
        return
    if not os.path.exists(dataset_path):
        print(f"エラー: データセットファイルが見つかりません: {dataset_path}", file=sys.stderr)
        return
    if not os.path.exists(classifier_model_path):
        print(f"エラー: CSV分類モデルが見つかりません: {classifier_model_path}", file=sys.stderr)
        print("CSV分類モデルを学習するには、csv_classifier_agent を train モードで実行してください。", file=sys.stderr)
        return

    # CSV分類モデルのロード
    try:
        classifier_model = joblib.load(classifier_model_path)
        print(f"CSV分類モデルをロードしました: {os.path.basename(classifier_model_path)}")
    except Exception as e:
        print(f"エラー: CSV分類モデルのロードに失敗しました: {e}", file=sys.stderr)
        return

    # データセットのタイプを予測
    print(f"データセット {os.path.basename(dataset_path)} のタイプを予測中...")
    features = analyze_csv_features(dataset_path)
    feature_vector = [features[key] for key in sorted(features.keys())]
    predicted_type = classifier_model.predict(np.array(feature_vector).reshape(1, -1))[0]
    print(f"予測されたデータセットタイプ: {predicted_type}")

    # 予測されたタイプに基づいて推奨事項を表示
    print("\n========== モデル推奨事項 ==========")
    if predicted_type == "mnist":
        print("このデータセットはMNISTのような分類タスクに適している可能性があります。")
        print("推奨モデルタイプ: 画像分類モデル (例: CNNベースのモデル)")
        print("考慮すべき前処理: 画像の正規化 (0-1スケール)、リサイズ、グレースケール変換、データ拡張。")
    elif predicted_type == "reinforce":
        print("このデータセットは強化学習のログデータである可能性があり、強化学習モデルの学習状況分析に適しています。")
        print("推奨モデルタイプ: 強化学習モデル (例: REINFORCE, DQN, A2Cなど)")
        print("考慮すべき前処理: 状態空間と行動空間の定義、報酬設計、エピソードの構造化。")
    else:
        print("このデータセットのタイプは現在のモデルでは特定できませんでした。")
        print("より多くのデータタイプでモデルを再学習することを検討してください。")
    print("====================================")

    print("Dataset Recommender Agent: 終了")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='データセット推奨エージェント')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_CONFIG["dataset_path"], help='分析したいデータセットのCSVファイルのパス')
    parser.add_argument('--classifier_model_path', type=str, default=DEFAULT_CONFIG["classifier_model_path"], help='CSV分類モデルのパス')
    args = parser.parse_args()
    
    config = {
        "dataset_path": args.dataset_path,
        "classifier_model_path": args.classifier_model_path
    }
    main([], config)
