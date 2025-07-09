# Yggdrasil Agent Framework

## 概要

Yggdrasil Agent Framework は、AI開発ワークフローの自動化と実験管理を効率化するために設計された汎用的なフレームワークです。データ前処理、モデル学習、評価といった一連のAIタスクをモジュール化された「エージェント」を通じて動的に実行・オーケストレーションします。

**本フレームワークは、Google Gemini CLIとの連携を前提として設計されており、Gemini CLIを使用することで、対話的にエージェントを実行し、実験をシームレスに管理できます。**

## Gemini CLIとの連携

本フレームワークは、Gemini CLIの強力な対話型インターフェースとコード実行能力を最大限に活用するように設計されています。以下の手順や使用例は、Gemini CLI環境での操作を前提としています。

Gemini CLIを使用することで、以下のメリットを享受できます。
*   **対話的な開発:** エージェントの実行、ログの確認、コードの修正などを対話的に行えます。
*   **効率的な実験管理:** 実験パラメータの調整や結果の追跡が容易になります。
*   **シームレスなワークフロー:** 開発から実験、レポート生成までの一連のプロセスをCLI上で完結できます。

## 主な機能

*   **エージェントの動的実行:** 独立したPythonスクリプトとして実装された「エージェント」を動的にロードし、実行します。
*   **AIワークフローのオーケストレーション:** `pipeline_orchestrator` エージェントを通じて、データ前処理、モデル学習、評価といった一連のAIワークフローを自動的に実行します。
*   **MLflow連携による実験管理:** 各エージェントの実行におけるパラメータ、メトリクス、生成されたモデルなどの情報を自動的に追跡・記録します（MLflow UIで視覚的に確認可能）。
*   **CSVタイプ分類:** `csv_classifier_agent` を使用して、CSVファイルの構造からそのタイプ（例: MNISTログ、強化学習ログ）を自動的に判別します。
*   **自動モデル評価:** `model_evaluator_agent` を使用して、学習済みモデルの性能を自動的に評価し、評価結果を記録します。
*   **最適なモデル選択:** `model_selector_agent` を使用して、評価ログから最適なモデルを自動的に選択します。
*   **ハイパーパラメータ最適化推奨:** `hyperparameter_optimizer_agent` を使用して、モデルの評価結果に基づいて次の学習に推奨されるハイパーパラメータを提案します。
*   **完全自動化ワークフロー:** `meta_trainer_agent` を通じて、「データ判別 → 学習 → 評価 → モデル選択 → ハイパーパラメータ最適化 → レポート生成」という一連のAI開発ワークフローを自動的に実行します。

## セットアップ

Yggdrasil Agent Framework を使用するためのセットアップ手順は以下の通りです。

### 1. リポジトリのクローン

まず、Yggdrasil Agent Framework のリポジトリをクローンします。

```bash
git clone <リポジトリのURL>
cd yggdrasil-agent-framework # クローンしたディレクトリに移動
```

### 2. 仮想環境の構築と依存関係のインストール

プロジェクトの依存関係を管理するために、Pythonの仮想環境を使用することを強く推奨します。以下のコマンドを実行して仮想環境を構築し、必要なライブラリをインストールします。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` ファイルには、プロジェクトに必要なすべての依存関係がリストされています。

### 3. Gemini CLIのインストールと設定

Gemini CLIのインストールと設定については、Gemini CLIの公式ドキュメントを参照してください。本フレームワークは、Gemini CLIがインストールされ、適切に設定されていることを前提としています。

## 基本的な使い方（Gemini CLIでの操作例）

フレームワークのメインスクリプト `yggdrasil.py` を使用して、任意のエージェントを実行できます。エージェントは、`agents/` ディレクトリに配置されたPythonファイルです。

**注意:** 以下のコマンドは、Gemini CLIのプロンプトから実行することを想定しています。

### エージェントの実行方法

```
# 例: hello_agent を実行
.venv/bin/python yggdrasil.py hello_agent
```

### パラメータの渡し方 (`--agent-set`)

エージェントにパラメータを渡すには、`--agent-set KEY=VALUE` オプションを使用します。

```
# 例: hello_agent に名前を指定して実行
.venv/bin/python yggdrasil.py hello_agent --agent-set name=Gemini
```

### `meta_trainer_agent` の実行例（完全自動化ワークフロー）

`meta_trainer_agent` を使用すると、データセットのタイプを自動判別し、適切なモデルの学習、評価、選択、ハイパーパラメータの推奨、そしてレポート生成までの一連のワークフローを自動的に実行できます。

```
# MNISTログファイルを使って完全自動化ワークフローを実行
.venv/bin/python yggdrasil.py meta_trainer_agent --agent-set data_file_path=logs/pipeline_experiment_log.csv

# 強化学習ログファイルを使って完全自動化ワークフローを実行
.venv/bin/python yggdrasil.py meta_trainer_agent --agent-set data_file_path=logs/reinforce_cartpole_log.csv
```

## 主要エージェント

*   `model_trainer`: 汎用的な学習スクリプトを実行し、モデルの学習と保存、および結果のロギングを行います。
*   `pipeline_orchestrator`: データ前処理、モデル学習、モデル評価といった一連のAIワークフローを自動的に実行します。
*   `report_generator_agent`: 実験ログ（CSV）を読み込み、論文風のMarkdownレポートを生成します。
*   `character_image_generator`: テキストから文字画像を生成し、データセットを作成します。
*   `csv_classifier_agent`: CSVファイルの構造からそのタイプを自動的に判別します。
*   `model_evaluator_agent`: 学習済みモデルの性能を自動的に評価し、評価結果を記録します。
*   `model_selector_agent`: 評価ログから最適なモデルを自動的に選択します。
*   `hyperparameter_optimizer_agent`: モデルの評価結果に基づいて次の学習に推奨されるハイパーパラメータを提案します。
*   `meta_trainer_agent`: データセットのタイプを自動判別し、適切な学習、評価、モデル選択、ハイパーパラメータ最適化、レポート生成までの一連のワークフローを自動的に実行します。

## MLflow連携

Yggdrasil Agent Framework は、MLflowと密接に連携し、AI実験の追跡と管理を容易にします。

### MLflow UIの起動方法

実験結果を視覚的に確認するには、MLflow UIを起動します。プロジェクトのルートディレクトリで、仮想環境をアクティベートしてから以下のコマンドをGemini CLIから実行します。

```
.venv/bin/mlflow ui
```

通常、MLflow UIは `http://localhost:5000` で利用可能になります。ウェブブラウザでこのURLにアクセスしてください。

## プロジェクト構造

```
yggdrasil-agent-framework/
├── .venv/                  # 仮想環境ディレクトリ
├── agents/                 # エージェント定義ファイル
│   ├── hello_agent.py
│   ├── model_trainer.py
│   ├── pipeline_orchestrator.py
│   ├── report_generator_agent.py
│   ├── character_image_generator.py
│   ├── csv_classifier_agent.py
│   ├── model_evaluator_agent.py
│   ├── model_selector_agent.py
│   ├── hyperparameter_optimizer_agent.py
│   └── meta_trainer_agent.py
├── config/                 # 設定ファイル
├── data/                   # データファイル (生成されたデータなど)
├── logs/                   # ログファイル (CSVログなど)
├── mlruns/                 # MLflowのトラッキングデータ
├── tests/                  # テストファイル
├── trained_models/         # 学習済みモデルの保存先
├── training_scripts/       # 学習およびデータ処理スクリプト
│   ├── data_preprocessor.py
│   ├── mnist_trainer.py
│   ├── model_evaluator.py
│   ├── character_recognizer.py
│   ├── csv_classifier_trainer.py
│   └── reinforce_cartpole_trainer.py
├── yggdrasil.py            # フレームワークのメインディスパッチャ
├── requirements.txt        # Pythonの依存関係リスト
├── GEMINI_RESEARCH_LOG.txt # 開発ログ
├── USER_GUIDE.md           # ユーザーガイド
└── README.md               # このファイル
```

## トラブルシューティング

Gemini CLIでの操作中に問題が発生した場合、以下の点を確認してください。

*   **仮想環境のアクティベート:** エージェントを実行する前に、必ず仮想環境がアクティベートされていることを確認してください。
*   **エラーメッセージの確認:** ターミナルに表示されるエラーメッセージの全文を確認し、原因を特定してください。
*   **ログファイルの確認:** `logs/` ディレクトリ内のログファイルに詳細な情報が記録されている場合があります。
*   **`USER_GUIDE.md` の参照:** より詳細な情報やトラブルシューティングのヒントについては、`USER_GUIDE.md` を参照してください。

## 貢献

本プロジェクトへの貢献を歓迎します。バグ報告、機能提案、プルリクエストなど、お気軽にお寄せください。