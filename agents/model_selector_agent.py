import os
import argparse
import csv
import sys

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# デフォルト設定
DEFAULT_CONFIG = {
    "evaluation_log_file": os.path.join(PROJECT_ROOT, "logs", "model_evaluation_log.csv"),
    "output_best_model_path": os.path.join(PROJECT_ROOT, "trained_models", "best_model.txt")
}

def main(args, config):
    """
    モデル評価ログを解析し、最適なモデルを選択するエージェント。
    """
    print("Model Selector Agent: 開始")

    evaluation_log_file = config.get("evaluation_log_log_file", DEFAULT_CONFIG["evaluation_log_file"])
    output_best_model_path = config.get("output_best_model_path", DEFAULT_CONFIG["output_best_model_path"])

    if not os.path.exists(evaluation_log_file):
        print(f"エラー: 評価ログファイルが見つかりません: {evaluation_log_file}", file=sys.stderr)
        return

    best_accuracy = -1.0
    best_model_path = None
    
    print(f"--- 評価ログを解析中: {evaluation_log_file} ---")
    try:
        with open(evaluation_log_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    accuracy = float(row.get('accuracy', 0.0))
                    model_path = row.get('model_path')

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_path = model_path
                except ValueError:
                    print(f"警告: 無効な精度値が検出されました。行をスキップします: {row}", file=sys.stderr)
                    continue

    except (IOError, csv.Error) as e:
        print(f"エラー: 評価ログファイルの読み込みまたは解析に失敗しました: {e}", file=sys.stderr)
        return

    if best_model_path:
        print(f"\n========== 最適なモデル ==========")
        print(f"パス: {best_model_path}")
        print(f"精度: {best_accuracy:.4f}")
        print(f"==============================")

        # 最適なモデルのパスをファイルに保存
        try:
            output_dir = os.path.dirname(output_best_model_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_best_model_path, 'w', encoding='utf-8') as f:
                f.write(best_model_path)
            print(f"最適なモデルのパスを保存しました: {output_best_model_path}")
        except IOError as e:
            print(f"エラー: 最適なモデルのパスの書き出しに失敗しました: {e}", file=sys.stderr)
    else:
        print("最適なモデルが見つかりませんでした。", file=sys.stderr)

    print("Model Selector Agent: 終了")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='モデル選択エージェント')
    parser.add_argument('--evaluation_log_file', type=str, default=DEFAULT_CONFIG["evaluation_log_file"], help='評価ログファイルのパス')
    parser.add_argument('--output_best_model_path', type=str, default=DEFAULT_CONFIG["output_best_model_path"], help='最適なモデルのパスの出力先')
    args = parser.parse_args()
    
    config = {
        "evaluation_log_file": args.evaluation_log_file,
        "output_best_model_path": args.output_best_model_path
    }
    main([], config)
