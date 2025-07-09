import os
from collections import Counter
from janome.tokenizer import Tokenizer

# プロジェクトのルートディレクトリを基準にパスを設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_UTTERANCES_FILE = os.path.join(PROJECT_ROOT, 'data', 'user_utterances.txt')

def analyze_utterances(top_n=20):
    """
    ユーザーの発言を形態素解析し、頻出単語を分析する。
    """
    print("ユーザーの発言傾向分析を開始します...")

    if not os.path.exists(USER_UTTERANCES_FILE):
        print(f"エラー: ユーザーの発言ファイルが見つかりません: {USER_UTTERANCES_FILE}")
        print("先に chat_log_parser.py を実行して、発言を抽出してください。")
        return

    t = Tokenizer()
    all_words = []

    try:
        with open(USER_UTTERANCES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # 形態素解析
                for token in t.tokenize(line.strip()):
                    # 品詞でフィルタリング（名詞、動詞、形容詞など）
                    # Janomeの品詞体系に合わせて調整
                    part_of_speech = token.part_of_speech.split(',')[0]
                    if part_of_speech in ['名詞', '動詞', '形容詞']:
                        all_words.append(token.base_form) # 基本形を使用
        print(f"ユーザーの発言 {USER_UTTERANCES_FILE} を読み込み、形態素解析しました。")
    except Exception as e:
        print(f"エラー: 発言ファイルの読み込みまたは解析中にエラーが発生しました: {e}")
        return

    if not all_words:
        print("警告: 解析できる単語が見つかりませんでした。")
        return

    # 単語の出現頻度をカウント
    word_counts = Counter(all_words)

    print(f"\n--- 頻出単語トップ {top_n} --- ")
    for word, count in word_counts.most_common(top_n):
        print(f'{word}: {count}')
    print("------------------------")

    print("ユーザーの発言傾向分析が完了しました。")

if __name__ == '__main__':
    analyze_utterances()
