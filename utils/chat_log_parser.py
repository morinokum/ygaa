

import os
import glob

# プロジェクトのルートディレクトリを基準にパスを設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_LOGS_DIR = os.path.join(PROJECT_ROOT, '..', 'my_gemini_project', 'chat_logs') # my_gemini_projectはyggdrasilの親ディレクトリにあると仮定
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'user_utterances.txt')

def parse_chat_logs():
    """
    チャットログからユーザーの発言を抽出し、一つのファイルにまとめる。
    """
    print("チャットログの解析を開始します...")

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    extracted_utterances = []
    log_files = glob.glob(os.path.join(CHAT_LOGS_DIR, 'gemini_chat_*.txt'))

    if not log_files:
        print(f"警告: チャットログファイルが見つかりません: {CHAT_LOGS_DIR}/gemini_chat_*.txt")
        print("ユーザーの発言抽出をスキップします。")
        return

    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('あなた:'):
                        # 'あなた:' プレフィックスを削除し、前後の空白を削除
                        utterance = line[len('あなた:'):].strip()
                        if utterance: # 空行は追加しない
                            extracted_utterances.append(utterance)
        except Exception as e:
            print(f"エラー: ファイル {log_file} の読み込み中にエラーが発生しました: {e}")
            continue

    if not extracted_utterances:
        print("警告: ログファイルからユーザーの発言が見つかりませんでした。")
        return

    # 抽出した発言をファイルに書き出す
    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            for utterance in extracted_utterances:
                f.write(utterance + '\n')
        print(f"ユーザーの発言を {OUTPUT_FILE_PATH} に抽出しました。")
    except Exception as e:
        print(f"エラー: 発言の書き出し中にエラーが発生しました: {e}")

    print("チャットログの解析が完了しました。")

if __name__ == '__main__':
    parse_chat_logs()

