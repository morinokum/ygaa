import subprocess
import sys
import os
import joblib
import numpy as np

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# utilsディレクトリをパスに追加
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
from csv_analyzer import analyze_csv_features

# Yggdrasilフレームワークのメインスクリプトのパス
YGGDDRASIL_MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "yggdrasil.py")

# 仮想環境のPythonインタプリタのパス
PYTHON_EXECUTABLE = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

# デフォルト設定
DEFAULT_CONFIG = {
    "data_file_path": None, # 学習させたいデータが含まれるCSVファイル
    "classifier_model_path": os.path.join(PROJECT_ROOT, "trained_models", "csv_classifier_model.joblib")
}

def run_agent(agent_name, agent_config):
    """
    指定されたエージェントを、設定を渡して実行するヘルパー関数。
    """
    command = [
        PYTHON_EXECUTABLE,
        YGGDDRASIL_MAIN_SCRIPT,
        agent_name,
    ]

    # エージェント固有の設定を --agent-set 形式で追加
    for key, value in agent_config.items():
        # パスは絶対パスに変換して渡す
        if "path" in key or "file" in key:
            if not os.path.isabs(str(value)):
                value = os.path.join(PROJECT_ROOT, value)
        command.extend(["--agent-set", f"{key}={value}"])

    print(f"\n--- エージェント '{agent_name}' を実行中 ---")
    print(f"実行コマンド: {' '.join(command)}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"エラー: エージェント '{agent_name}' の実行に失敗しました。リターンコード: {e.returncode}", file=sys.stderr)
        if e.stdout:
            print("--- stdout/stderr ---", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
        sys.exit(1) # パイプラインを停止
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1) # パイプラインを停止

def main(args, config):
    """
    CSVファイルのタイプを判別し、適切な学習エージェントを呼び出すメタトレーナーエージェント。
    """
    print("Meta Trainer Agent: 開始")

    data_file_path = config.get("data_file_path", DEFAULT_CONFIG["data_file_path"])
    classifier_model_path = config.get("classifier_model_path", DEFAULT_CONFIG["classifier_model_path"])

    if not data_file_path:
        print("エラー: 学習させたいデータファイルが指定されていません。--agent-set data_file_path=<path_to_csv> で指定してください。", file=sys.stderr)
        return
    if not os.path.exists(data_file_path):
        print(f"エラー: データファイルが見つかりません: {data_file_path}", file=sys.stderr)
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

    # データファイルのタイプを予測
    print(f"データファイル {os.path.basename(data_file_path)} のタイプを予測中...")
    features = analyze_csv_features(data_file_path)
    feature_vector = [features[key] for key in sorted(features.keys())]
    predicted_log_type = classifier_model.predict(np.array(feature_vector).reshape(1, -1))[0]
    print(f"予測されたデータタイプ: {predicted_log_type}")

    # 予測されたタイプに基づいて適切な学習エージェントを呼び出す
    if predicted_log_type == "mnist":
        print("MNISTデータタイプを検出しました。MNISTモデルの学習を開始します。")
        mnist_trainer_config = {
            "script_path": "training_scripts/mnist_trainer.py",
            "output_path": os.path.join(PROJECT_ROOT, "trained_models", "mnist_model_from_meta.keras"),
            "log_file": os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log.csv"), # 既存のログファイルに追記
            "epochs": 5, # デフォルトのエポック数
            "batch_size": 32, # デフォルトのバッチサイズ
            "learning_rate": 0.001 # デフォルトの学習率
        }
        run_agent("model_trainer", mnist_trainer_config)

    elif predicted_log_type == "reinforce":
        print("強化学習データタイプを検出しました。CartPole強化学習モデルの学習を開始します。")
        reinforce_trainer_config = {
            "output_path": os.path.join(PROJECT_ROOT, "trained_models", "reinforce_model_from_meta.keras"),
            "log_file": os.path.join(PROJECT_ROOT, "logs", "reinforce_cartpole_log.csv"), # 既存のログファイルに追記
            "episodes": 500, # デフォルトのエピソード数
            "learning_rate": 0.001, # デフォルトの学習率
            "gamma": 0.99 # デフォルトの割引率
        }
        run_agent("reinforcement_learner", reinforce_trainer_config)

    else:
        print(f"警告: 未知のデータタイプ '{predicted_log_type}' が予測されました。学習エージェントは呼び出されません。", file=sys.stderr)

    print("Meta Trainer Agent: 終了")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CSVタイプに基づいて学習エージェントを呼び出すメタトレーナーエージェント')
    parser.add_argument('--data_file_path', type=str, default=DEFAULT_CONFIG["data_file_path"], help='学習させたいデータが含まれるCSVファイルのパス')
    parser.add_argument('--classifier_model_path', type=str, default=DEFAULT_CONFIG["classifier_model_path"], help='CSV分類モデルのパス')
    args = parser.parse_args()
    
    config = {
        "data_file_path": args.data_file_path,
        "classifier_model_path": args.classifier_model_path
    }
    main([], config)
