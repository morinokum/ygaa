import csv
import os

def is_int(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def analyze_csv_features(file_path):
    features = {
        'num_columns': 0,
        'has_timestamp': False,
        'has_epochs': False,
        'has_batch_size': False,
        'has_learning_rate': False,
        'has_test_loss': False,
        'has_test_accuracy': False,
        'has_episodes': False,
        'has_gamma': False,
        'has_avg_last_100_rewards': False,
        'has_solved': False,
        'can_convert_epochs_to_int': False,
        'can_convert_batch_size_to_int': False,
        'can_convert_learning_rate_to_float': False,
        'can_convert_test_loss_to_float': False,
        'can_convert_test_accuracy_to_float': False,
        'can_convert_episodes_to_int': False,
        'can_convert_gamma_to_float': False,
        'can_convert_avg_last_100_rewards_to_float': False,
    }

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) # ヘッダーを読み込む
            features['num_columns'] = len(header)

            # ヘッダーの存在チェック
            if 'timestamp' in header: features['has_timestamp'] = True
            if 'epochs' in header: features['has_epochs'] = True
            if 'batch_size' in header: features['has_batch_size'] = True
            if 'learning_rate' in header: features['has_learning_rate'] = True
            if 'test_loss' in header: features['has_test_loss'] = True
            if 'test_accuracy' in header: features['has_test_accuracy'] = True
            if 'episodes' in header: features['has_episodes'] = True
            if 'gamma' in header: features['has_gamma'] = True
            if 'avg_last_100_rewards' in header: features['has_avg_last_100_rewards'] = True
            if 'solved' in header: features['has_solved'] = True

            # データの型変換可能性チェック (最初の数行で十分)
            sample_rows = []
            for i, row in enumerate(reader):
                if i >= 10: # 最初の10行をサンプリング
                    break
                if len(row) == len(header):
                    sample_rows.append(dict(zip(header, row)))
            
            for row_dict in sample_rows:
                if features['has_epochs'] and is_int(row_dict.get('epochs')): features['can_convert_epochs_to_int'] = True
                if features['has_batch_size'] and is_int(row_dict.get('batch_size')): features['can_convert_batch_size_to_int'] = True
                if features['has_learning_rate'] and is_float(row_dict.get('learning_rate')): features['can_convert_learning_rate_to_float'] = True
                if features['has_test_loss'] and is_float(row_dict.get('test_loss')): features['can_convert_test_loss_to_float'] = True
                if features['has_test_accuracy'] and is_float(row_dict.get('test_accuracy')): features['can_convert_test_accuracy_to_float'] = True
                if features['has_episodes'] and is_int(row_dict.get('episodes')): features['can_convert_episodes_to_int'] = True
                if features['has_gamma'] and is_float(row_dict.get('gamma')): features['can_convert_gamma_to_float'] = True
                if features['has_avg_last_100_rewards'] and is_float(row_dict.get('avg_last_100_rewards')): features['can_convert_avg_last_100_rewards_to_float'] = True

    except (IOError, csv.Error) as e:
        print(f"エラー: CSVファイルの読み込みまたは解析に失敗しました: {e}", file=sys.stderr)
        # エラーが発生した場合でも、部分的に抽出できた特徴を返す
    return features

if __name__ == '__main__':
    # テスト用
    # 既存のログファイルパス
    mnist_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "pipeline_experiment_log.csv")
    reinforce_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "reinforce_cartpole_log.csv")

    print(f"--- Analyzing MNIST Log: {mnist_log_path} ---")
    mnist_features = analyze_csv_features(mnist_log_path)
    for key, value in mnist_features.items():
        print(f"  {key}: {value}")

    print(f"\n--- Analyzing Reinforce Log: {reinforce_log_path} ---")
    reinforce_features = analyze_csv_features(reinforce_log_path)
    for key, value in reinforce_features.items():
        print(f"  {key}: {value}")
