import os
import argparse
import csv
import sys
import numpy as np
import json

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# デフォルト設定
DEFAULT_CONFIG = {
    "evaluation_log_file": os.path.join(PROJECT_ROOT, "logs", "model_evaluation_log.csv"),
    "target_accuracy": 0.98, # 目標精度
    "max_epochs_increase": 10, # エポック数を増やす最大値
    "learning_rate_factors": [0.5, 1.0, 2.0], # 学習率を調整する係数
    "output_json": False # JSON形式で出力を生成するかどうか
}

def main(args, config):
    """
    モデル評価ログを解析し、最適なハイパーパラメータを推奨するエージェント。
    """
    evaluation_log_file = config.get("evaluation_log_file", DEFAULT_CONFIG["evaluation_log_file"])
    target_accuracy = config.get("target_accuracy", DEFAULT_CONFIG["target_accuracy"])
    max_epochs_increase = config.get("max_epochs_increase", DEFAULT_CONFIG["max_epochs_increase"])
    learning_rate_factors = config.get("learning_rate_factors", DEFAULT_CONFIG["learning_rate_factors"])
    output_json = config.get("output_json", DEFAULT_CONFIG["output_json"])

    if not output_json:
        print("Hyperparameter Optimizer Agent: 開始")

    if not os.path.exists(evaluation_log_file):
        if output_json:
            print(json.dumps({"error": f"評価ログファイルが見つかりません: {evaluation_log_file}"}), file=sys.stderr)
        else:
            print(f"エラー: 評価ログファイルが見つかりません: {evaluation_log_file}", file=sys.stderr)
        return

    best_accuracy = -1.0
    best_run_params = {}
    
    if not output_json:
        print(f"--- 評価ログを解析中: {evaluation_log_file} ---")
    try:
        with open(evaluation_log_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # accuracyが数値で、かつN/Aでないことを確認
                    if row.get('accuracy') and row['accuracy'] != 'N/A':
                        accuracy = float(row['accuracy'])
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_run_params = row
                except ValueError:
                    if not output_json:
                        print(f"警告: 無効な精度値が検出されました。行をスキップします: {row}", file=sys.stderr)
                    continue

    except (IOError, csv.Error) as e:
        if output_json:
            print(json.dumps({"error": f"評価ログファイルの読み込みまたは解析に失敗しました: {e}"}), file=sys.stderr)
        else:
            print(f"エラー: 評価ログファイルの読み込みまたは解析に失敗しました: {e}", file=sys.stderr)
        return

    recommendation = {}

    if best_accuracy >= target_accuracy:
        recommendation["status"] = "target_achieved"
        recommendation["message"] = "目標精度に到達しました。さらなる最適化は不要かもしれません。"
    elif not best_run_params:
        recommendation["status"] = "no_valid_data"
        recommendation["message"] = "有効な評価データが見つかりませんでした。初期学習から開始してください。"
    else:
        recommendation["status"] = "recommendation_available"
        recommendation["message"] = "次の学習のためのハイパーパラメータを推奨します。"
        
        # 現在の最良のハイパーパラメータを取得
        current_epochs = int(best_run_params.get('epochs', 5)) if best_run_params.get('epochs', '').isdigit() else 5
        current_batch_size = int(best_run_params.get('batch_size', 32)) if best_run_params.get('batch_size', '').isdigit() else 32
        current_learning_rate = float(best_run_params.get('learning_rate', 0.001)) if best_run_params.get('learning_rate', '').replace('.', '', 1).isdigit() else 0.001

        # 新しいハイパーパラメータの提案 (シンプルなヒューリスティック)
        suggested_epochs = current_epochs + max_epochs_increase
        suggested_learning_rate = current_learning_rate * np.random.choice(learning_rate_factors)
        
        # learning_rateが極端に小さくならないように下限を設定
        if suggested_learning_rate < 1e-5: 
            suggested_learning_rate = 1e-5

        recommendation["recommended_parameters"] = {
            "epochs": suggested_epochs,
            "batch_size": current_batch_size,
            "learning_rate": round(suggested_learning_rate, 6) # 小数点以下6桁に丸める
        }

    if output_json:
        print(json.dumps(recommendation))
    else:
        print(f"現在の最高精度: {best_accuracy:.4f}")
        if recommendation["status"] == "target_achieved":
            print(recommendation["message"])
        elif recommendation["status"] == "no_valid_data":
            print(recommendation["message"])
        else:
            print("--- 次の学習のためのハイパーパラメータを推奨中 ---")
            print(f"推奨エポック数: {recommendation["recommended_parameters"]["epochs"]}")
            print(f"推奨学習率: {recommendation["recommended_parameters"]["learning_rate"]:.6f}")
            print(f"バッチサイズ: {recommendation["recommended_parameters"]["batch_size"]} (変更なし)")

            # meta_trainer_agent を実行するためのコマンドを生成
            # ここではMNISTのログを想定しているため、pipeline_experiment_log.csv を使用
            recommended_command = (
                f"{os.path.join(PROJECT_ROOT, ".venv", "bin", "python")} "
                f"{os.path.join(PROJECT_ROOT, "yggdrasil.py")} meta_trainer_agent "
                f"--agent-set data_file_path={os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log.csv")} "
                f"--agent-set epochs={recommendation["recommended_parameters"]["epochs"]} "
                f"--agent-set batch_size={recommendation["recommended_parameters"]["batch_size"]} "
                f"--agent-set learning_rate={recommendation["recommended_parameters"]["learning_rate"]:.6f}"
            )
            print("\nMeta Trainer Agent を実行するための推奨コマンド:")
            print(recommended_command)

    if not output_json:
        print("Hyperparameter Optimizer Agent: 終了")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ハイパーパラメータ最適化エージェント')
    parser.add_argument('--evaluation_log_file', type=str, default=DEFAULT_CONFIG["evaluation_log_file"], help='評価ログファイルのパス')
    parser.add_argument('--target_accuracy', type=float, default=DEFAULT_CONFIG["target_accuracy"], help='目標精度')
    parser.add_argument('--max_epochs_increase', type=int, default=DEFAULT_CONFIG["max_epochs_increase"], help='エポック数を増やす最大値')
    parser.add_argument('--learning_rate_factors', nargs='+', type=float, default=DEFAULT_CONFIG["learning_rate_factors"], help='学習率を調整する係数のリスト')
    parser.add_argument('--output_json', action='store_true', help='結果をJSON形式で出力する')
    args = parser.parse_args()
    
    config = {
        "evaluation_log_file": args.evaluation_log_file,
        "target_accuracy": args.target_accuracy,
        "max_epochs_increase": args.max_epochs_increase,
        "learning_rate_factors": args.learning_rate_factors,
        "output_json": args.output_json
    }
    main([], config)