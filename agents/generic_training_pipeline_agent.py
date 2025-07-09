import subprocess
import sys
import os
import argparse

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 仮想環境のPythonインタプリタのパス
PYTHON_EXECUTABLE = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

# Yggdrasilフレームワークのメインスクリプトのパス
YGGDDRASIL_MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "yggdrasil.py")

# デフォルト設定 (必要に応じて拡張)
DEFAULT_CONFIG = {
    "model_type": "", # 例: "image_classification", "text_classification", "tabular_classification"
    "training_script_path": "", # 訓練スクリプトのパス
    "evaluation_script_path": "", # 評価スクリプトのパス
    "dataset_path": "", # データセットのパス
    "output_model_path": os.path.join(PROJECT_ROOT, "trained_models", "generic_model.keras"),
    "log_file": os.path.join(PROJECT_ROOT, "logs", "generic_pipeline_log.csv"),
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer_type": "adam"
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

    for key, value in agent_config.items():
        # パスは絶対パスに変換して渡す
        if "path" in key or "file" in key:
            if value and not os.path.isabs(str(value)):
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
    汎用モデル訓練パイプラインをオーケストレーションするエージェント。
    """
    print("Generic Training Pipeline Agent: 開始")

    # 設定の取得 (コマンドライン引数やデフォルト設定から)
    model_type = config.get("model_type", DEFAULT_CONFIG["model_type"])
    training_script_path = config.get("training_script_path", DEFAULT_CONFIG["training_script_path"])
    evaluation_script_path = config.get("evaluation_script_path", DEFAULT_CONFIG["evaluation_script_path"])
    dataset_path = config.get("dataset_path", DEFAULT_CONFIG["dataset_path"])
    output_model_path = config.get("output_model_path", DEFAULT_CONFIG["output_model_path"])
    print(f"Debug: output_model_path (absolute) = {os.path.join(PROJECT_ROOT, output_model_path) if not os.path.isabs(output_model_path) else output_model_path}")
    log_file = config.get("log_file", DEFAULT_CONFIG["log_file"])
    epochs = config.get("epochs", DEFAULT_CONFIG["epochs"])
    batch_size = config.get("batch_size", DEFAULT_CONFIG["batch_size"])
    learning_rate = config.get("learning_rate", DEFAULT_CONFIG["learning_rate"])
    optimizer_type = config.get("optimizer_type", DEFAULT_CONFIG["optimizer_type"])

    # 1. モデル訓練ステップ
    if training_script_path:
        train_config = {
            "script_path": training_script_path,
            "dataset_path": dataset_path,
            "output_path": output_model_path,
            "log_file": log_file,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer_type": optimizer_type
        }
        # model_trainer エージェントは script_path を特別扱いするため、直接渡す
        run_agent("model_trainer", {"script_path": training_script_path, **train_config})
    else:
        print("警告: 訓練スクリプトのパスが指定されていないため、訓練ステップをスキップします。")

    # 2. モデル評価ステップは generic_trainer.py 内で行われるため、ここではスキップ
    print("評価ステップは訓練スクリプト内で実行されました。")

    # 3. レポート生成ステップ (必要に応じて)
    # report_generator_agent を呼び出すロジックをここに追加することも可能

    print("Generic Training Pipeline Agent: 終了")

if __name__ == '__main__':
    # このスクリプトが直接実行された場合のテスト用
    # yggdrasil.pyから呼び出される場合は、main関数に引数が渡される
    parser = argparse.ArgumentParser(description="Generic Training Pipeline Agent (for direct testing).")
    parser.add_argument("--model_type", type=str, default=DEFAULT_CONFIG["model_type"])
    parser.add_argument("--training_script_path", type=str, default=DEFAULT_CONFIG["training_script_path"])
    parser.add_argument("--evaluation_script_path", type=str, default=DEFAULT_CONFIG["evaluation_script_path"])
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_CONFIG["dataset_path"])
    parser.add_argument("--output_model_path", type=str, default=DEFAULT_CONFIG["output_model_path"])
    parser.add_argument("--log_file", type=str, default=DEFAULT_CONFIG["log_file"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--optimizer_type", type=str, default=DEFAULT_CONFIG["optimizer_type"])

    cli_args = parser.parse_args()

    # argparseのNamespaceオブジェクトを辞書に変換
    config_from_cli = vars(cli_args)

    main([], config_from_cli)
