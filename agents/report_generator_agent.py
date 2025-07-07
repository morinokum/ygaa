import os
import csv
from datetime import datetime

# デフォルト設定
DEFAULT_CONFIG = {
    "log_file_path": "logs/pipeline_experiment_log.csv",
    "report_output_path": "Experiment_Report.md"
}

def analyze_results(data):
    """
    実験結果データを簡易的に分析し、考察を生成する。
    """
    if len(data) < 2:
        return "実験データが不足しているため、詳細な考察はできません。"

    # 簡単な比較のため、最後の2つの実験結果を取得
    last_run = data[-1]
    previous_run = data[-2]

    # キーが存在するか確認し、なければデフォルト値を設定
    last_accuracy = float(last_run.get('test_accuracy', 0))
    prev_accuracy = float(previous_run.get('test_accuracy', 0))
    last_epochs = int(last_run.get('epochs', 0))
    prev_epochs = int(previous_run.get('epochs', 0))

    insight = "### 考察\n\n"
    if last_accuracy > prev_accuracy:
        insight += f"- **精度の向上:** 直近の実験 (エポック数: {last_epochs}) では、その前の実験 (エポック数: {prev_epochs}) と比較して、テスト精度が {prev_accuracy:.4f} から {last_accuracy:.4f} に向上しました。これは、パラメータ変更がモデル性能に良い影響を与えた可能性を示唆しています。\n"
    elif last_accuracy < prev_accuracy:
        insight += f"- **精度の低下:** 直近の実験 (エポック数: {last_epochs}) では、その前の実験 (エポック数: {prev_epochs}) と比較して、テスト精度が {prev_accuracy:.4f} から {last_accuracy:.4f} に低下しました。パラメータ設定の見直しが必要かもしれません。\n"
    else:
        insight += "- **精度の変化なし:** 直近の実験と前の実験でテスト精度に変化は見られませんでした。\n"
    
    insight += "- **今後の課題:** さらなる精度向上のためには、ハイパーパラメータ（学習率、オプティマイザなど）の調整や、モデルアーキテクチャの変更を検討することが有効と考えられます。\n"
    return insight

def main(args, config):
    """
    実験ログ(CSV)を読み込み、論文風のMarkdownレポートを生成するエージェント。
    """
    print("Report Generator Agent: 開始")

    # 設定の取得
    log_file_path = config.get("log_file_path", DEFAULT_CONFIG["log_file_path"])
    report_output_path = config.get("report_output_path", DEFAULT_CONFIG["report_output_path"])
    
    # プロジェクトルートからの絶対パスに変換
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_abs_path = os.path.join(project_root, log_file_path)
    report_output_abs_path = os.path.join(project_root, report_output_path)

    if not os.path.exists(log_file_abs_path):
        print(f"エラー: ログファイルが見つかりません: {log_file_abs_path}", file=sys.stderr)
        return

    # CSVデータの読み込み
    header = []
    data = []
    try:
        with open(log_file_abs_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) # ヘッダーを読み込む
            for row in reader:
                if len(row) == 6: # 列数が6の行（学習ステップのログ）のみを処理
                    row_dict = dict(zip(header, row))
                    data.append(row_dict)

    except (IOError, csv.Error) as e:
        print(f"エラー: ログファイルの読み込みに失敗しました: {e}", file=sys.stderr)
        return

    # レポートの生成
    print(f"--- レポートを生成中: {report_output_path} ---")
    
    report_content = f"""
# 実験レポート

**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 概要

本レポートは、Yggdrasil Agent Framework を用いて実施された一連のAIモデル学習実験の結果をまとめたものである。
目的は、異なるハイパーパラメータ（エポック数、バッチサイズ）がMNISTデータセットに対するモデルの性能（精度、損失）に与える影響を定量的に評価することである。

## 2. 実験環境

- **フレームワーク:** Yggdrasil Agent Framework
- **学習タスク:** MNIST手書き数字画像の分類
- **使用モデル:** TensorFlow/Kerasで実装されたシンプルなニューラルネットワーク
- **評価指標:** テストデータセットに対する損失（categorical_crossentropy）および精度（accuracy）

## 3. 実験結果

以下に、実施された実験のパラメータと結果を示す。

| 実行日時 | エポック数 | バッチサイズ | テスト損失 | テスト精度 |
|---|---|---|---|---|
"""
    # 結果の表を作成
    for row in data:
        # キーの存在を確認
        timestamp = row.get('timestamp', 'N/A')
        epochs = row.get('epochs', 'N/A')
        batch_size = row.get('batch_size', 'N/A')
        test_loss = row.get('test_loss', 'N/A')
        test_accuracy = row.get('test_accuracy', 'N/A')
        report_content += f"| {timestamp} | {epochs} | {batch_size} | {test_loss} | {test_accuracy} |\n"

    report_content += "\n"
    
    # 考察の追加
    report_content += analyze_results(data)
    
    report_content += """
## 4. 結論

本一連の実験により、ハイパーパラメータの変更がモデル性能に与える影響を確認した。
今回の結果に基づき、今後はさらなるパラメータチューニングや、より複雑なモデル構造の検討を進めることが推奨される。
"""

    # レポートをファイルに書き出し
    try:
        with open(report_output_abs_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"--- レポートの生成が完了しました: {report_output_path} ---")
    except IOError as e:
        print(f"エラー: レポートファイルの書き出しに失敗しました: {e}", file=sys.stderr)

    print("Report Generator Agent: 終了")

if __name__ == '__main__':
    # このスクリプトが直接実行された場合のテスト用
    main([], DEFAULT_CONFIG)
