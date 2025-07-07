
import subprocess
import sys
import os

# このエージェントファイルの場所を基準にプロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 仮想環境のPythonインタプリタのパス
PYTHON_EXECUTABLE = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

# Yggdrasilフレームワークのメインスクリプトのパス
YGGDDRASIL_MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "yggdrasil.py")

# デフォルト設定
DEFAULT_CONFIG = {
    "processed_data_path": os.path.join(PROJECT_ROOT, "data", "processed_data.npz"),
    "trained_model_path": os.path.join(PROJECT_ROOT, "trained_models", "pipeline_model.keras"),
    "experiment_log_file": os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log.csv"),
    "epochs": 1, # パイプラインでのデフォルトエポック数
    "batch_size": 32, # パイプラインでのデフォルトバッチサイズ
    "learning_rate": 0.001, # デフォルトの学習率
    "optimizer_type": "adam" # デフォルトのオプティマイザ
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
    # script_path は特別扱いし、それ以外の引数を渡す
    script_to_run = agent_config["script_path"]
    agent_args = {
        k: v for k, v in agent_config.items() if k != "script_path"
    }

    command.extend(["--agent-set", f"script_path={script_to_run}"])

    for key, value in agent_args.items():
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
    AIワークフローパイプラインをオーケストレーションするエージェント。
    """
    print("Pipeline Orchestrator Agent: 開始")

    # 設定の取得
    processed_data_path = config.get("processed_data_path", DEFAULT_CONFIG["processed_data_path"])
    trained_model_path = config.get("trained_model_path", DEFAULT_CONFIG["trained_model_path"])
    experiment_log_file = config.get("experiment_log_file", DEFAULT_CONFIG["experiment_log_file"])
    epochs = config.get("epochs", DEFAULT_CONFIG["epochs"])
    batch_size = config.get("batch_size", DEFAULT_CONFIG["batch_size"])
    learning_rate = config.get("learning_rate", DEFAULT_CONFIG["learning_rate"])
    optimizer_type = config.get("optimizer_type", DEFAULT_CONFIG["optimizer_type"])

    # 1. 文字認識モデル学習ステップ
    character_recognizer_config = {
        "script_path": "training_scripts/character_recognizer.py",
        "output_path": trained_model_path,
        "log_file": experiment_log_file,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_type": optimizer_type
    }
    run_agent("model_trainer", character_recognizer_config)

    # 2. モデル評価ステップ
    model_evaluator_config = {
        "script_path": "training_scripts/model_evaluator.py",
        "model_path": trained_model_path,
        "input_data_path": os.path.join(PROJECT_ROOT, "data", "neo_world_characters.npz"), # 評価にも同じデータセットのテスト部分を使用
        "log_file": experiment_log_file # 学習ログと同じファイルに追記
    }
    run_agent("model_trainer", model_evaluator_config)

    print("Pipeline Orchestrator Agent: 終了")

if __name__ == '__main__':
    # このスクリプトが直接実行された場合のテスト用
    main([], DEFAULT_CONFIG)
