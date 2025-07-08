import os
import csv
from datetime import datetime
import sys
import joblib
import numpy as np

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# utilsディレクトリをパスに追加
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
from csv_analyzer import analyze_csv_features

# デフォルト設定
DEFAULT_CONFIG = {
    "log_file_path": os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log_migrated.csv"),
    "report_output_path": os.path.join(PROJECT_ROOT, "Experiment_Report.md"),
    "classifier_model_path": os.path.join(PROJECT_ROOT, "trained_models", "csv_classifier_model.joblib")
}

def load_and_filter_data(log_file_path, log_type):
    """
    指定されたログファイルからデータを読み込み、ログタイプに基づいてフィルタリングする。
    """
    data = []
    try:
        with open(log_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) # ヘッダーを読み込む
            for row in reader:
                row_dict = dict(zip(header, row))
                
                if log_type == "mnist":
                    # MNISTログの条件: epochsとbatch_sizeが数値で、test_lossとtest_accuracyが存在する
                    if row_dict.get('epochs', '').isdigit() and \
                       row_dict.get('batch_size', '').isdigit() and \
                       'test_loss' in row_dict and 'test_accuracy' in row_dict:
                        data.append(row_dict)
                elif log_type == "reinforce":
                    # 強化学習ログの条件: avg_last_100_rewardsが存在する
                    if 'avg_last_100_rewards' in row_dict:
                        data.append(row_dict)
    except (IOError, csv.Error) as e:
        print(f"エラー: ログファイルの読み込みまたは解析に失敗しました: {e}", file=sys.stderr)
    return data

def generate_markdown_report(data, log_type, report_output_path):
    """
    フィルタリングされたデータからMarkdownレポートを生成する。
    """
    sections = {
        "mnist": {
            "title": "MNISTモデル学習実験レポート",
            "overview": "本レポートは、Yggdrasil Agent Framework を用いて実施された一連のAIモデル学習実験の結果をまとめたものである。目的は、異なるハイパーパラメータ（エポック数、バッチサイズ、学習率）がMNISTデータセットに対するモデルの性能（精度、損失）に与える影響を定量的に評価することである。",
            "environment": "- **フレームワーク:** Yggdrasil Agent Framework\n- **学習タスク:** MNIST手書き数字画像の分類\n- **使用モデル:** TensorFlow/Kerasで実装されたシンプルなニューラルネットワーク\n- **評価指標:** テストデータセットに対する損失（categorical_crossentropy）および精度（accuracy）",
            "table_header": "| 実行日時 | エポック数 | バッチサイズ | 学習率 | テスト損失 | テスト精度 |\n|---|---|---|---|---|---|\n",
            "row_format": lambda row: f"| {row.get('timestamp', 'N/A')} | {row.get('epochs', 'N/A')} | {row.get('batch_size', 'N/A')} | {row.get('learning_rate', 'N/A') if 'learning_rate' in row and row['learning_rate'] != 'N/A' else 'N/A'} | {row.get('test_loss', 'N/A')} | {row.get('test_accuracy', 'N/A')} |\n",
            "conclusion": "本一連の実験により、ハイパーパラメータの変更がモデル性能に与える影響を確認した。今回の結果に基づき、今後はさらなるパラメータチューニングや、より複雑なモデル構造の検討を進めることが推奨される。"
        },
        "reinforce": {
            "title": "CartPole強化学習実験レポート",
            "overview": "本レポートは、Yggdrasil Agent Framework を用いて実施されたCartPole環境における強化学習実験の結果をまとめたものである。目的は、REINFORCEアルゴリズムを用いたエージェントの学習状況を評価することである。",
            "environment": "- **フレームワーク:** Yggdrasil Agent Framework\n- **学習タスク:** CartPole-v1 環境における強化学習\n- **使用アルゴリズム:** REINFORCE\n- **評価指標:** エピソードごとの累積報酬、過去100エピソードの平均報酬",
            "table_header": "| 実行日時 | エピソード数 | 学習率 | 割引率 | 過去100エピソード平均報酬 | 環境解決 |\n|---|---|---|---|---|---|\n",
            "row_format": lambda row: f"| {row.get('timestamp', 'N/A')} | {row.get('episodes', 'N/A')} | {row.get('learning_rate', 'N/A')} | {row.get('gamma', 'N/A')} | {row.get('avg_last_100_rewards', 'N/A')} | {row.get('solved', 'N/A')} |\n",
            "conclusion": "本一連の実験により、REINFORCEアルゴリズムを用いたCartPole環境の学習状況を確認した。今回の結果に基づき、今後はさらなるアルゴリズムの改善や、より複雑な環境への適用を検討することが推奨される。"
        }
    }

    current_sections = sections.get(log_type)
    if current_sections is None:
        print(f"エラー: 未知のログタイプが指定されました: {log_type}", file=sys.stderr)
        return

    report_content = f"""
# {current_sections["title"]}

**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 概要

{current_sections["overview"]}

## 2. 実験環境

{current_sections["environment"]}

## 3. 実験結果

以下に、実施された実験のパラメータと結果を示す。

{current_sections["table_header"]}
"""

    for row in data:
        report_content += current_sections["row_format"](row)

    report_content += "\n"
    
    report_content += analyze_results(data, log_type)
    
    report_content += f"""
## 4. 結論

{current_sections["conclusion"]}
"""

    try:
        with open(report_output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"--- レポートの生成が完了しました: {os.path.basename(report_output_path)} ---")
    except IOError as e:
        print(f"エラー: レポートファイルの書き出しに失敗しました: {e}", file=sys.stderr)

def analyze_results(data, log_type):
    insight = "### 考察\n\n"
    if log_type == "mnist":
        if len(data) < 2:
            insight += "実験データが不足しているため、詳細な考察はできません。\n"
        else:
            last_run = data[-1]
            previous_run = data[-2]

            # 数値変換を試みる前に、値が存在し、かつ数値として解釈可能かを確認
            last_accuracy = float(last_run.get('test_accuracy', 0)) if last_run.get('test_accuracy', '').replace('.', '', 1).isdigit() else 0.0
            prev_accuracy = float(previous_run.get('test_accuracy', 0)) if previous_run.get('test_accuracy', '').replace('.', '', 1).isdigit() else 0.0
            last_epochs = int(last_run.get('epochs', 0)) if last_run.get('epochs', '').isdigit() else 0
            prev_epochs = int(previous_run.get('epochs', 0)) if previous_run.get('epochs', '').isdigit() else 0
            
            # 表示用の学習率。数値変換は行わない
            last_lr_display = last_run.get('learning_rate', 'N/A')
            prev_lr_display = previous_run.get('learning_rate', 'N/A')

            if last_accuracy > prev_accuracy:
                insight += f"- **精度の向上:** 直近の実験 (エポック数: {last_epochs}, 学習率: {last_lr_display}) では、その前の実験 (エポック数: {prev_epochs}, 学習率: {prev_lr_display}) と比較して、テスト精度が {prev_accuracy:.4f} から {last_accuracy:.4f} に向上しました。これは、パラメータ変更がモデル性能に良い影響を与えた可能性を示唆しています。\n"
            elif last_accuracy < prev_accuracy:
                insight += f"- **精度の低下:** 直近の実験 (エポック数: {last_epochs}, 学習率: {last_lr_display}) では、その前の実験 (エポック数: {prev_epochs}, 学習率: {prev_lr_display}) と比較して、テスト精度が {prev_accuracy:.4f} から {last_accuracy:.4f} に低下しました。パラメータ設定の見直しが必要かもしれません。\n"
            else:
                insight += "- **精度の変化なし:** 直近の実験と前の実験でテスト精度に変化は見られませんでした。\n"
        insight += "- **今後の課題:** さらなる精度向上のためには、ハイパーパラメータ（学習率、オプティマイザなど）の調整や、モデルアーキテクチャの変更を検討することが有効と考えられます。\n"

    elif log_type == "reinforce":
        if len(data) < 1:
            insight += "実験データが不足しているため、詳細な考察はできません。\n"
        else:
            last_run = data[-1]
            solved = last_run.get('solved', 'No')
            avg_reward = float(last_run.get('avg_last_100_rewards', 0)) if last_run.get('avg_last_100_rewards', '').replace('.', '', 1).isdigit() else 0.0
            episodes_run = int(last_run.get('episodes', 0)) if last_run.get('episodes', '').isdigit() else 0

            insight += f"- **学習状況:** 最新の実験では、{episodes_run} エピソードを実行し、過去100エピソードの平均報酬は {avg_reward:.2f} でした。環境は{'解決済み' if solved == 'Yes' else '未解決'}です。\n"
            if solved == 'Yes':
                insight += "- **成功:** 環境の解決基準を満たしました。エージェントはCartPole環境で安定してバランスを取ることを学習しました。\n"
            else:
                insight += "- **課題:** 環境の解決基準には達していません。さらなる学習エピソードの追加、ハイパーパラメータ（学習率、割引率）の調整、または異なる強化学習アルゴリズムの検討が必要かもしれません。\n"
        insight += "- **今後の課題:** さらなる性能向上のためには、報酬設計の見直し、より複雑なポリシーネットワークの検討、または他の強化学習アルゴリズム（例: DQN, A2C）の導入が有効と考えられます。\n"

    return insight

def main(args, config):
    """
    実験ログ(CSV)を読み込み、論文風のMarkdownレポートを生成するエージェント。
    """
    print("Report Generator Agent: 開始")

    # 設定の取得
    log_file_path = config.get("log_file_path", DEFAULT_CONFIG["log_file_path"])
    report_output_path = config.get("report_output_path", DEFAULT_CONFIG["report_output_path"])
    classifier_model_path = config.get("classifier_model_path", DEFAULT_CONFIG["classifier_model_path"])
    
    # プロジェクトルートからの絶対パスに変換 (DEFAULT_CONFIGで既に絶対パスになっているが念のため)
    if not os.path.isabs(log_file_path):
        log_file_abs_path = os.path.join(PROJECT_ROOT, log_file_path)
    else:
        log_file_abs_path = log_file_path

    if not os.path.isabs(report_output_path):
        report_output_abs_path = os.path.join(PROJECT_ROOT, report_output_path)
    else:
        report_output_abs_path = report_output_path

    if not os.path.isabs(classifier_model_path):
        classifier_model_abs_path = os.path.join(PROJECT_ROOT, classifier_model_path)
    else:
        classifier_model_abs_path = classifier_model_path

    if not os.path.exists(classifier_model_abs_path):
        print(f"エラー: CSV分類モデルが見つかりません: {classifier_model_abs_path}", file=sys.stderr)
        print("CSV分類モデルを学習するには、csv_classifier_agent を train モードで実行してください。", file=sys.stderr)
        return
    try:
        classifier_model = joblib.load(classifier_model_abs_path)
        print(f"CSV分類モデルをロードしました: {os.path.basename(classifier_model_abs_path)}")
    except Exception as e:
        print(f"エラー: CSV分類モデルのロードに失敗しました: {e}", file=sys.stderr)
        return

    # ログファイルのタイプを予測
    print(f"ログファイル {os.path.basename(log_file_abs_path)} のタイプを予測中...")
    features = analyze_csv_features(log_file_abs_path)
    feature_vector = [features[key] for key in sorted(features.keys())]
    predicted_log_type = classifier_model.predict(np.array(feature_vector).reshape(1, -1))[0]
    print(f"予測されたログタイプ: {predicted_log_type}")

    # CSVデータの読み込みとフィルタリング
    data = load_and_filter_data(log_file_abs_path, predicted_log_type)
    if not data:
        print(f"警告: ログファイル {log_file_abs_path} から有効なデータが見つかりませんでした。レポートは生成されません。", file=sys.stderr)
        return

    # レポートの生成と書き出し
    generate_markdown_report(data, predicted_log_type, report_output_abs_path)

    print("Report Generator Agent: 終了")

if __name__ == '__main__':
    # このスクリプトが直接実行された場合のテスト用
    import argparse
    parser = argparse.ArgumentParser(description='実験レポート生成エージェント')
    parser.add_argument('--log_file_path', type=str, default=DEFAULT_CONFIG["log_file_path"], help='入力ログファイルのパス')
    parser.add_argument('--report_output_path', type=str, default=DEFAULT_CONFIG["report_output_path"], help='生成されるレポートの出力パス')
    # log_type は自動判別されるため、ここでは削除
    args = parser.parse_args()
    
    config = {
        "log_file_path": args.log_file_path,
        "report_output_path": args.report_output_path,
        "classifier_model_path": DEFAULT_CONFIG["classifier_model_path"]
    }
    main([], config)
