import os
import subprocess
import sys
import json

# プロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YGGDDRASIL_MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "yggdrasil.py")
PYTHON_EXECUTABLE = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
MLFLOW_EXECUTABLE = os.path.join(PROJECT_ROOT, ".venv", "bin", "mlflow")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log.csv")
REPORT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Experiment_Report.md")

def run_command(command, cwd=PROJECT_ROOT):
    """シェルコマンドを実行し、出力をリアルタイムで表示する"""
    print(f"\n--- コマンド実行中: {' '.join(command)} ---")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd)
        for line in process.stdout:
            sys.stdout.write(line)
        process.wait()
        if process.returncode != 0:
            print(f"エラー: コマンドが非ゼロ終了コードで終了しました: {process.returncode}", file=sys.stderr)
    except FileNotFoundError:
        print(f"エラー: コマンドが見つかりません。パスを確認してください: {command[0]}", file=sys.stderr)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
    print(f"--- コマンド実行完了 ---")

def select_agent():
    """利用可能なエージェントを選択するメニューを表示する"""
    agents_dir = os.path.join(PROJECT_ROOT, "agents")
    agent_files = [f for f in os.listdir(agents_dir) if f.endswith(".py") and f != "__init__.py"]
    agents = sorted([os.path.splitext(f)[0] for f in agent_files])

    print("\n--- エージェント選択 ---")
    for i, agent in enumerate(agents):
        print(f"{i+1}. {agent}")
    print("--------------------")

    while True:
        try:
            choice = int(input("実行するエージェントの番号を入力してください: "))
            if 1 <= choice <= len(agents):
                return agents[choice-1]
            else:
                print("無効な番号です。")
        except ValueError:
            print("無効な入力です。数字を入力してください。")

def get_agent_parameters(agent_name):
    """エージェントに応じたパラメータを対話的に取得する"""
    params = {}
    if agent_name == "pipeline_orchestrator":
        print("\n--- パイプラインオーケストレーターのパラメータ ---")
        epochs = input("エポック数 (デフォルト: 1): ")
        params["epochs"] = int(epochs) if epochs else 1
        batch_size = input("バッチサイズ (デフォルト: 32): ")
        params["batch_size"] = int(batch_size) if batch_size else 32
        learning_rate = input("学習率 (デフォルト: 0.001): ")
        params["learning_rate"] = float(learning_rate) if learning_rate else 0.001
        optimizer_type = input("オプティマイザ (デフォルト: adam, 例: sgd): ")
        params["optimizer_type"] = optimizer_type if optimizer_type else "adam"
    elif agent_name == "hello_agent":
        name = input("名前 (デフォルト: World): ")
        if name:
            params["name"] = name
    # 他のエージェントのパラメータもここに追加
    return params

def main_menu():
    """メインメニューを表示する"""
    while True:
        print("\n--- Yggdrasil Agent Framework CLI ---")
        print("1. エージェントを実行する")
        print("2. MLflow UIを起動する")
        print("3. 実験レポートを生成・表示する")
        print("4. Gitプッシュを支援する")
        print("5. 終了")
        print("-----------------------------------")

        choice = input("選択してください (1-5): ")

        if choice == "1":
            agent_name = select_agent()
            if agent_name:
                params = get_agent_parameters(agent_name)
                command = [PYTHON_EXECUTABLE, YGGDDRASIL_MAIN_SCRIPT, agent_name]
                for key, value in params.items():
                    command.extend(["--agent-set", f"{key}={value}"])
                run_command(command)
        elif choice == "2":
            run_command([MLFLOW_EXECUTABLE, "ui"])
        elif choice == "3":
            run_command([PYTHON_EXECUTABLE, YGGDDRASIL_MAIN_SCRIPT, "report_generator_agent"])
            if os.path.exists(REPORT_OUTPUT_PATH):
                print("\n--- 実験レポートの内容 ---")
                with open(REPORT_OUTPUT_PATH, "r", encoding="utf-8") as f:
                    print(f.read())
                print("------------------------")
            else:
                print("レポートファイルが見つかりません。")
        elif choice == "4":
            run_command([PYTHON_EXECUTABLE, os.path.join(PROJECT_ROOT, "cli_app", "git_push_helper_agent.py")])
        elif choice == "5":
            print("Yggdrasil Agent Framework CLIを終了します。")
            break
        else:
            print("無効な選択です。1から5の数字を入力してください。")

if __name__ == "__main__":
    main_menu()
