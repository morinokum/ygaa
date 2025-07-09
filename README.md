# Yggdrasil Agent Framework

## 概要

Yggdrasil Agent Framework は、AI開発ワークフローの自動化と実験管理を効率化するために設計された汎用的なフレームワークです。データ前処理、モデル学習、評価といった一連のAIタスクをモジュール化された「エージェント」を通じて動的に実行・オーケストレーションします。

**本フレームワークは、Google Gemini CLIとの連携を前提として設計されており、Gemini CLIを使用することで、対話的にエージェントを実行し、実験をシームレスに管理できます。**

### エージェントの整理と機能の絞り込み

本フレームワークは、モデル学習と評価の核となる機能に焦点を当てるため、エージェントの整理を行いました。主要な機能を提供するエージェントは`agents/`ディレクトリに配置され、より専門的またはユーティリティ的なエージェントは`agents/utilities/`ディレクトリにアーカイブされています。これにより、開発の効率化とフレームワークの目的の明確化を図っています。

## Gemini CLIとの連携

本フレームワークは、Gemini CLIの強力な対話型インターフェースとコード実行能力を最大限に活用するように設計されています。以下の手順や使用例は、Gemini CLI環境での操作を前提としています。

Gemini CLIを使用することで、以下のメリットを享受できます。
*   **対話的な開発:** エージェントの実行、ログの確認、コードの修正などを対話的に行えます。
*   **効率的な実験管理:** 実験パラメータの調整や結果の追跡が容易になります。
*   **シームレスなワークフロー:** 開発から実験、レポート生成までの一連のプロセスをCLI上で完結できます。

## 主な機能

*   **エージェントの動的実行:** 独立したPythonスクリプトとして実装された「エージェント」を動的にロードし、実行します。
*   **汎用モデル訓練パイプライン:** `generic_training_pipeline_agent` を使用して、様々な種類のモデル（画像分類、テキスト分類、表形式データなど）の訓練、評価、レポート生成を、設定可能なデータセットやハイパーパラメータで行うことができます。
*   **MLflow連携による実験管理:** 各エージェントの実行におけるパラメータ、メトリクス、生成されたモデルなどの情報を自動的に追跡・記録します（MLflow UIで視覚的に確認可能）。

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
# 例: generic_training_pipeline_agent を実行
.venv/bin/python yggdrasil.py generic_training_pipeline_agent
```

### パラメータの渡し方 (`--agent-set`)

エージェントにパラメータを渡すには、`--agent-set KEY=VALUE` オプションを使用します。

```
# 例: generic_training_pipeline_agent に訓練スクリプトとデータセットを指定して実行
.venv/bin/python yggdrasil.py generic_training_pipeline_agent --agent-set training_script_path=training_scripts/generic_trainer.py --agent-set dataset_path=data/my_dataset.csv
```

## 主要エージェント

*   `generic_training_pipeline_agent`: 汎用的なモデル訓練パイプラインをオーケストレーションします。
*   `model_trainer`: 汎用的な学習スクリプトを実行し、モデルの学習と保存、および結果のロギングを行います。
*   `model_evaluator_agent`: 学習済みモデルの性能を自動的に評価し、評価結果を記録します。

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
├── agents/                 # 主要エージェント定義ファイル
│   ├── generic_training_pipeline_agent.py
│   ├── model_trainer.py
│   └── model_evaluator_agent.py
├── agents/utilities/       # その他のエージェント（アーカイブ）
│   ├── csv_classifier_agent.py
│   ├── dataset_recommender_agent.py
│   ├── hello_agent.py
│   ├── hyperparameter_optimizer_agent.py
│   ├── inference_agent.py
│   ├── manage_agents.py
│   ├── meta_trainer_agent.py
│   ├── model_collection_agent.py
│   ├── model_selector_agent.py
│   ├── personal_context_agent.py
│   ├── pipeline_orchestrator.py
│   ├── reinforcement_learner.py
│   ├── report_generator_agent.py
│   ├── system_health_check_agent.py
│   └── topic_classifier_agent.py
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
│   ├── reinforce_cartpole_trainer.py
│   └── simple_regression.py
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