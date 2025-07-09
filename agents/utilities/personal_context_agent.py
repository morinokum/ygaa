import json
import datetime
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from janome.tokenizer import Tokenizer # 追加

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths
LOGS_DIR = "logs/user_logs"
DATA_DIR = "data"
USER_PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")
SUMMARY_FOR_GEMINI_PATH = os.path.join(DATA_DIR, "summary_for_gemini.txt")
USER_UTTERANCES_FILE = os.path.join(PROJECT_ROOT, 'data', 'user_utterances.txt') # 追加

def get_log_file_path(date_obj):
    """Generates the log file path for a given date."""
    return os.path.join(LOGS_DIR, date_obj.strftime("%Y%m%d_user_log.json"))

def load_json_file(file_path, default_value=None):
    """Loads a JSON file, returning default_value if not found or invalid."""
    if not os.path.exists(file_path):
        return default_value
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Returning default.")
        return default_value
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return default_value

def save_json_file(file_path, data):
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def tokenize_japanese(text):
    """
    A very basic tokenizer for Japanese text using regex.
    Splits text by common delimiters and extracts sequences of Japanese characters.
    This is a simplification for TF-IDF without a proper morphological analyzer.
    """
    # Split by spaces, punctuation, etc.
    tokens = re.findall(r'[ぁ-んァ-ヶー一-龠々]+|[a-zA-Z0-9]+', text)
    return " ".join(tokens) # TfidfVectorizer expects space-separated words

def main(agent_args):
    print("personal_context_agent started.")

    # 1. Handle user_message and log it
    user_message = agent_args.get("user_message")
    current_date = datetime.date.today()
    current_log_path = get_log_file_path(current_date)

    current_log_data = load_json_file(current_log_path, [])
    if user_message:
        timestamp = datetime.datetime.now().isoformat()
        current_log_data.append({"timestamp": timestamp, "message": user_message})
        save_json_file(current_log_path, current_log_data)
        print(f"Logged user message to {current_log_path}")

    # 2. Read previous day's log
    yesterday = current_date - datetime.timedelta(days=1)
    yesterday_log_path = get_log_file_path(yesterday)
    yesterday_log_data = load_json_file(yesterday_log_path, [])

    all_messages_yesterday = [entry["message"] for entry in yesterday_log_data if "message" in entry]
    if not all_messages_yesterday:
        print(f"No messages found in yesterday's log: {yesterday_log_path}. Skipping keyword extraction and profile update.")
        # Still proceed to generate summary, but it will be based on existing profile or empty.
        # If no previous log, the summary will be minimal.
        # We should still generate a summary for Gemini, even if it's just a "no new context" message.
        # For now, I'll let the rest of the code run, which will use an empty text for TF-IDF.
        # This means the profile won't be updated based on yesterday's log if it's empty.
        pass

    # 3. Keyword extraction (TF-IDF)
    # Prepare text for TF-IDF
    processed_text_for_tfidf = [tokenize_japanese(msg) for msg in all_messages_yesterday]
    
    keywords = []
    if processed_text_for_tfidf:
        try:
            vectorizer = TfidfVectorizer(min_df=1, stop_words=None) # min_df=1 to include all words
            tfidf_matrix = vectorizer.fit_transform(processed_text_for_tfidf)
            feature_names = vectorizer.get_feature_names_out()

            # Get top keywords from the combined text
            # Sum TF-IDF scores for each word across all documents
            sums = tfidf_matrix.sum(axis=0)
            ranking = [(feature_names[col], sums[0, col]) for col in sums.argsort()[0, ::-1]]
            
            # Extract top N keywords (e.g., top 10)
            keywords = [word for word, score in ranking[:10]]
            print(f"Extracted keywords from yesterday's log: {keywords}")
        except Exception as e:
            print(f"Error during TF-IDF keyword extraction: {e}")
            keywords = [] # Fallback to empty keywords

    # --- Start: Analyze all user utterances for overall frequent words ---
    overall_frequent_words = {}
    if os.path.exists(USER_UTTERANCES_FILE):
        t = Tokenizer()
        all_words_from_all_logs = []
        try:
            with open(USER_UTTERANCES_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    for token in t.tokenize(line.strip()):
                        part_of_speech = token.part_of_speech.split(',')[0]
                        if part_of_speech in ['名詞', '動詞', '形容詞']:
                            all_words_from_all_logs.append(token.base_form)
            print(f"ユーザーの発言全体 ({USER_UTTERANCES_FILE}) を読み込み、形態素解析しました。")
            if all_words_from_all_logs:
                overall_frequent_words = dict(Counter(all_words_from_all_logs).most_common(50)) # 上位50件
                print(f"全体からの頻出単語を抽出しました (上位50件)。")
            else:
                print("警告: ユーザーの発言全体から解析できる単語が見つかりませんでした。")
        except Exception as e:
            print(f"エラー: ユーザーの発言全体ファイルの読み込みまたは解析中にエラーが発生しました: {e}")
    else:
        print(f"警告: ユーザーの発言全体ファイルが見つかりません: {USER_UTTERANCES_FILE}")
    # --- End: Analyze all user utterances for overall frequent words ---

    # 4. Update user_profile.json
    user_profile = load_json_file(USER_PROFILE_PATH, {
        "frequent_words": {},
        "topic_trends": [],
        "usage_style": {},
        "top_frequent_words_from_all_logs": {} # 新しいキーを追加
    })

    # Update frequent words (simple count for now, can be refined with TF-IDF scores)
    if all_messages_yesterday:
        combined_text = " ".join(all_messages_yesterday)
        words = re.findall(r'[ぁ-んァ-ヶー一-龠々]+|[a-zA-Z0-9]+', combined_text)
        word_counts = Counter(words)
        
        # Merge with existing frequent words, keeping top N
        for word, count in word_counts.items():
            user_profile["frequent_words"][word] = user_profile["frequent_words"].get(word, 0) + count
        
        # Keep only top 50 frequent words
        user_profile["frequent_words"] = dict(sorted(user_profile["frequent_words"].items(), key=lambda item: item[1], reverse=True)[:50])

    # Update topic trends with extracted keywords
    if keywords:
        # Add new keywords to topic trends, avoiding duplicates and keeping recent ones
        for kw in keywords:
            if kw not in user_profile["topic_trends"]:
                user_profile["topic_trends"].append(kw)
        # Keep only the last N (e.g., 20) topic trends
        user_profile["topic_trends"] = user_profile["topic_trends"][-20:]

    # Update usage style (e.g., average message length, can add more metrics)
    if all_messages_yesterday:
        total_length = sum(len(msg) for msg in all_messages_yesterday)
        avg_length = total_length / len(all_messages_yesterday) if all_messages_yesterday else 0
        user_profile["usage_style"]["avg_message_length_yesterday"] = round(avg_length, 2)
        # You can add more metrics here, e.g., sentiment analysis if a library is available.

    # Update overall frequent words from all logs
    if overall_frequent_words: # overall_frequent_words は前のステップで計算済み
        user_profile["top_frequent_words_from_all_logs"] = overall_frequent_words

    save_json_file(USER_PROFILE_PATH, user_profile)
    print(f"Updated user profile at {USER_PROFILE_PATH}")

    # 5. Format for Gemini CLI and output to summary_for_gemini.txt
    summary_text = "--- Personal Context Summary for Gemini CLI ---\n\n" \
                   "ユーザーの最近の関心事:\n"
    if user_profile["topic_trends"]:
        summary_text += "・" + "、".join(user_profile["topic_trends"]) + "\n"
    else:
        summary_text += "（特になし）\n"

    summary_text += "\nユーザーの頻出語:\n"
    if user_profile["frequent_words"]:
        top_frequent = dict(sorted(user_profile["frequent_words"].items(), key=lambda item: item[1], reverse=True)[:5])
        summary_text += "・" + "、".join([f"{word} ({count})" for word, count in top_frequent.items()]) + "\n"
    else:
        summary_text += "（特になし）\n"

    summary_text += "\nユーザーのメッセージスタイル:\n"
    if "avg_message_length_yesterday" in user_profile["usage_style"]:
        summary_text += f"・昨日の平均メッセージ長: {user_profile['usage_style']['avg_message_length_yesterday']} 文字\n"
    else:
        summary_text += "（情報なし）\n"

    summary_text += "\n--- End of Summary ---\n"

    with open(SUMMARY_FOR_GEMINI_PATH, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Generated summary for Gemini CLI at {SUMMARY_FOR_GEMINI_PATH}")

    print("personal_context_agent finished.")

if __name__ == "__main__":
    # This block is for direct testing, not for yggdrasil.py execution
    # Example usage:
    # main({"user_message": "今日はAIの倫理について考えていた"})
    # main({}) # To process yesterday's log without new message
    print("This agent is designed to be run via yggdrasil.py.")
    print("Example: python yggdrasil.py personal_context_agent --agent-set user_message=\"今日の天気は晴れです\"")