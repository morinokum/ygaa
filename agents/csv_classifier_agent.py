import subprocess
import sys
import os
import joblib
import numpy as np

# このエージェントファイルの場所を基準にプロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# utilsディレクトリをパスに追加
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
from csv_analyzer import analyze_csv_features

# デフォルト設定
DEFAULT_CONFIG = {
    "mode": "train", # 'train' または 'predict'
    "trainer_script_path": os.path.join(PROJECT_ROOT, "training_scripts", "csv_classifier_trainer.py"),
    "model_path": os.path.join(PROJECT_ROOT, "trained_models", "csv_classifier_model.joblib"),
    "target_csv_path": None # predictモードで使用
}

def main(args, config):
    """
    CSV分類モデルの学習または推論を実行するエージェント。
    """
    print(f"CSV Classifier Agent: 開始 (モード: {config['mode']})")

    mode = config.get("mode", DEFAULT_CONFIG["mode"])
    model_path = config.get("model_path", DEFAULT_CONFIG["model_path"])

    if mode == "train":
        trainer_script_path = config.get("trainer_script_path", DEFAULT_CONFIG["trainer_script_path"])
        if not os.path.exists(trainer_script_path):
            print(f"エラー: トレーナースクリプトが見つかりません: {trainer_script_path}", file=sys.stderr)
            return

        python_executable = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
        command = [
            python_executable,
            trainer_script_path,
            "--output_model_path", model_path
        ]

        try:
            print(f"実行コマンド: {' '.join(command)}")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

        except subprocess.CalledProcessError as e:
            print(f"エラー: トレーナースクリプトの実行に失敗しました。リターンコード: {e.returncode}", file=sys.stderr)
            if e.stdout:
                print("--- stdout/stderr ---", file=sys.stderr)
                print(e.stdout, file=sys.stderr)
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)

    elif mode == "predict":
        target_csv_path = config.get("target_csv_path", DEFAULT_CONFIG["target_csv_path"])
        if not target_csv_path:
            print("エラー: 予測するCSVファイルが指定されていません。--agent-set target_csv_path=<path_to_csv> で指定してください。", file=sys.stderr)
            return
        if not os.path.exists(target_csv_path):
            print(f"エラー: CSVファイルが見つかりません: {target_csv_path}", file=sys.stderr)
            return
        if not os.path.exists(model_path):
            print(f"エラー: 学習済みモデルが見つかりません: {model_path}", file=sys.stderr)
            return

        # モデルのロード
        try:
            model = joblib.load(model_path)
            print(f"モデルをロードしました: {model_path}")
        except Exception as e:
            print(f"エラー: モデルのロードに失敗しました: {e}", file=sys.stderr)
            return

        # CSVファイルから特徴量を抽出
        print(f"CSVファイルから特徴量を抽出中: {target_csv_path}")
        features = analyze_csv_features(target_csv_path)
        feature_vector = [features[key] for key in sorted(features.keys())]
        X_predict = np.array(feature_vector).reshape(1, -1) # 1サンプルとして整形

        # 予測を実行
        prediction = model.predict(X_predict)
        print(f"\n========== 予測結果 ==========")
        print(f"CSVファイル: {os.path.basename(target_csv_path)}")
        print(f"予測されたタイプ: {prediction[0]}")
        print(f"==============================")

    else:
        print(f"エラー: 未知のモードが指定されました: {mode}", file=sys.stderr)

    print("CSV Classifier Agent: 終了")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CSV分類エージェント')
    parser.add_argument('--mode', type=str, default=DEFAULT_CONFIG["mode"], choices=['train', 'predict'], help='実行モード (train または predict)')
    parser.add_argument('--trainer_script_path', type=str, default=DEFAULT_CONFIG["trainer_script_path"], help='トレーナースクリプトのパス (trainモードで使用)')
    parser.add_argument('--model_path', type=str, default=DEFAULT_CONFIG["model_path"], help='学習済みモデルのパス')
    parser.add_argument('--target_csv_path', type=str, default=DEFAULT_CONFIG["target_csv_path"], help='予測するCSVファイルのパス (predictモードで使用)')
    args = parser.parse_args()
    
    config = {
        "mode": args.mode,
        "trainer_script_path": args.trainer_script_path,
        "model_path": args.model_path,
        "target_csv_path": args.target_csv_path
    }
    main([], config)
