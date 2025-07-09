

import joblib
import os
import sys

# プロジェクトのルートディレクトリを基準にパスを設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'topic_classifier.joblib')

# デフォルト設定
DEFAULT_CONFIG = {
    "text": None # 分類したいテキスト
}

def main(args, config):
    """
    学習済みモデルをロードし、与えられたテキストのトピックを分類するエージェント。
    """
    print("Topic Classifier Agent: 開始")

    text_to_classify = config.get("text", DEFAULT_CONFIG["text"])

    if not text_to_classify:
        print("エラー: 分類するテキストが指定されていません。--agent-set text=\"your text\" で指定してください。", file=sys.stderr)
        return

    # 1. モデルのロード
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"学習済みモデルをロードしました: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"エラー: 学習済みモデルが見つかりません: {MODEL_PATH}", file=sys.stderr)
        print("モデルを学習するには、training_scripts/topic_model_trainer.py を実行してください。", file=sys.stderr)
        return
    except Exception as e:
        print(f"エラー: モデルのロード中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        return

    # 2. テキストの分類
    print(f'入力テキスト: "{text_to_classify}"')
    
    # モデルはリスト形式の入力を期待するため、テキストをリストに入れる
    input_text_list = [text_to_classify]
    
    predicted_topic = pipeline.predict(input_text_list)[0]
    predicted_probabilities = pipeline.predict_proba(input_text_list)
    
    # 確率をトピック名とセットで表示
    classes = pipeline.classes_
    probabilities_dict = dict(zip(classes, predicted_probabilities[0]))

    print(f'\n予測されたトピック: {predicted_topic}')
    print('各トピックの予測確率:')
    for topic, prob in sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True):
        print(f'  - {topic}: {prob:.4f}')

    print("\nTopic Classifier Agent: 終了")

if __name__ == '__main__':
    # このエージェントは yggdrasil.py 経由での実行を想定
    print("このエージェントは yggdrasil.py 経由で実行してください。")
    print("例: python yggdrasil.py topic_classifier_agent --agent-set text=\"AIモデルの学習について知りたい\"")

