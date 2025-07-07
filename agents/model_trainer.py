
import subprocess
import sys
import os

# このエージェントファイルの場所を基準にプロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# デフォルト設定
DEFAULT_CONFIG = {
    "script_path": os.path.join(PROJECT_ROOT, "training_scripts", "mnist_trainer.py"),
}

def main(args, config):
    """
    汎用的なモデル学習スクリプトを実行するエージェント。
    """
    print("Model Trainer Agent: 開始")

    # 実行する学習スクリプトのパスを取得
    script_path = config.get("script_path", DEFAULT_CONFIG.get("script_path"))
    if not script_path:
        print("エラー: 実行する学習スクリプトのパスが指定されていません。", file=sys.stderr)
        return

    # script_pathが相対パスの場合は、プロジェクトルートからの絶対パスに変換
    if not os.path.isabs(script_path):
        script_path = os.path.join(PROJECT_ROOT, script_path)

    if not os.path.exists(script_path):
        print(f"エラー: 学習スクリプトが見つかりません: {script_path}", file=sys.stderr)
        return

    # 仮想環境のPythonインタプリタを使用
    python_executable = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

    command = [
        python_executable,
        script_path,
    ]

    # config内のすべてのパラメータをコマンドライン引数として追加
    for key, value in config.items():
        # script_path は既に処理済みなのでスキップ
        if key == "script_path":
            continue
        
        # data_preprocessor.py には parent_run_id を渡さない
        if "data_preprocessor.py" in script_path and key == "parent_run_id":
            continue

        # valueがNoneの場合は引数として渡さない
        if value is None:
            continue

        # パスやファイル名を含む引数は絶対パスに変換
        if "path" in key or "file" in key:
            if not os.path.isabs(str(value)):
                value = os.path.join(PROJECT_ROOT, value)
        
        command.extend([f"--{key}", str(value)])

    try:
        # 学習スクリプトをサブプロセスとして実行
        print(f"実行コマンド: {' '.join(command)}")
        
        # Popenを使用してリアルタイムで出力を取得
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # 出力をリアルタイムで表示
        for line in process.stdout:
            sys.stdout.write(line)
        
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"エラー: 学習スクリプトの実行に失敗しました。リターンコード: {e.returncode}", file=sys.stderr)
        if e.stdout:
            print("--- stdout/stderr ---", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)

    print("Model Trainer Agent: 終了")

if __name__ == '__main__':
    # このスクリプトが直接実行された場合のテスト用
    main([], DEFAULT_CONFIG)
