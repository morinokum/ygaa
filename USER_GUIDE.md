# Yggdrasil Agent Framework ユーザーガイド

## はじめに

Yggdrasil Agent Framework は、AI開発ワークフローの自動化と実験管理を効率化するために設計された汎用的なフレームワークです。このフレームワークは、モジュール化された「エージェント」を通じて、データ前処理、モデル学習、評価といった一連のAIタスクを動的に実行・オーケストレーションすることを可能にします。

### 目的と解決する課題

現代のAI開発は、データ準備、モデルの選択と学習、ハイパーパラメータチューニング、評価、デプロイといった多岐にわたるステップを含みます。これらのステップはしばしば手動で行われ、再現性の確保や実験の追跡が困難になることがあります。Yggdrasil Agent Framework は、以下の課題を解決することを目指します。

*   **ワークフローの自動化:** AI開発の各ステップを自動化し、手動での介入を最小限に抑えます。
*   **実験の再現性:** 各実験のパラメータ、コード、結果を体系的に記録し、再現可能な開発プロセスを確立します。
*   **実験管理の効率化:** MLflowとの統合により、実験の追跡、比較、分析を容易にします。
*   **モジュール性と拡張性:** 新しいタスクやモデルに容易に対応できるよう、柔軟なエージェントベースのアーキテクチャを提供します。

このガイドは、Yggdrasil Agent Framework のセットアップ、使用方法、および拡張方法について説明します。

## 主な機能

Yggdrasil Agent Framework は、AI開発プロセスを効率化するための以下の主要な機能を提供します。

### エージェントの動的実行

フレームワークは、独立したPythonスクリプトとして実装された「エージェント」を動的にロードし、実行する機能を提供します。これにより、特定のタスク（例: データ前処理、モデル学習、評価）を実行するエージェントを柔軟に組み合わせて使用できます。

### AIワークフローのオーケストレーション

`pipeline_orchestrator` エージェントを使用することで、複数のエージェントを連携させ、一連のAIワークフロー（例: データ生成 → モデル学習 → モデル評価）を自動的に実行できます。これにより、複雑な開発パイプラインを簡単に構築し、管理できます。

### MLflow連携による実験管理

MLflowとの統合により、各エージェントの実行（特にモデル学習や評価）におけるパラメータ、メトリクス、生成されたモデルなどの情報を自動的に追跡・記録します。MLflow UIを使用することで、実験結果を視覚的に比較・分析し、再現性を高めることができます。

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

`requirements.txt` ファイルには、プロジェクトに必要なすべての依存関係がリストされています。このコマンドを実行することで、それらが仮想環境にインストールされます。

**注意:** `requirements.txt` が存在しない場合は、プロジェクトのルートディレクトリに作成し、必要なライブラリ（例: `tensorflow`, `mlflow`, `numpy`, `scikit-learn` など）を記述してください。

## 基本的な使い方

Yggdrasil Agent Framework の基本的な使い方を説明します。

### エージェントの実行方法

フレームワークのメインスクリプト `yggdrasil.py` を使用して、任意のエージェントを実行できます。エージェントは、`agents/` ディレクトリに配置されたPythonファイルです。

```bash
python yggdrasil.py <エージェント名> [オプション]
```

例: `hello_agent` を実行する

```bash
python yggdrasil.py hello_agent
```

### 設定の渡し方 (`--set`, `--agent-set`)

エージェントにパラメータを渡すには、`--set` または `--agent-set` オプションを使用します。

*   `--set KEY=VALUE`: フレームワーク全体の設定を上書きします。これは、`yggdrasil.py` が読み込むデフォルト設定や、エージェントの設定ファイル（存在する場合）に影響を与えます。
*   `--agent-set KEY=VALUE`: 実行する特定のエージェントにのみパラメータを渡します。これは、エージェントの `main` 関数に `config` 辞書として渡されます。

例: `model_trainer` エージェントにエポック数を指定して実行する

```bash
python yggdrasil.py model_trainer --agent-set epochs=10
```

### `hello_agent` の実行例

`hello_agent` は、フレームワークの動作を確認するためのシンプルなエージェントです。

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# hello_agent を実行
python yggdrasil.py hello_agent

# 名前を指定して実行
python yggdrasil.py hello_agent --agent-set name=Gemini
```

## 主要エージェント

Yggdrasil Agent Framework には、AIワークフローの主要なタスクを実行するためのエージェントが用意されています。

### `model_trainer` の使い方と例

`model_trainer` エージェントは、汎用的な学習スクリプトを実行し、モデルの学習と保存、および結果のロギングを行います。内部的には、`training_scripts/` ディレクトリ内のスクリプト（例: `mnist_trainer.py`, `simple_regression.py`）を呼び出します。

**主なパラメータ:**

*   `script_path`: 実行する学習スクリプトのパス（`training_scripts/` からの相対パス）。
*   `input_data_path`: 入力データファイルへのパス。
*   `output_path`: 学習済みモデルの保存先パス。
*   `epochs`: 学習のエポック数。
*   `batch_size`: 学習のバッチサイズ。

**実行例:**

```bash
# MNISTモデルを学習し、結果をMLflowに記録
python yggdrasil.py model_trainer \
    --agent-set script_path=training_scripts/mnist_trainer.py \
    --agent-set input_data_path=data/processed_data.npz \
    --agent-set output_path=trained_models/my_mnist_model.keras \
    --agent-set epochs=5 \
    --agent-set batch_size=64
```

### `pipeline_orchestrator` の使い方と例

`pipeline_orchestrator` エージェントは、データ前処理、モデル学習、モデル評価といった一連のAIワークフローを自動的に実行します。このエージェントは、内部で他のエージェント（`model_trainer` など）を呼び出します。

**主なパラメータ:**

*   `processed_data_path`: 前処理済みデータの保存先パス。
*   `trained_model_path`: 学習済みモデルの保存先パス。
*   `epochs`: 学習ステップに渡すエポック数。
*   `batch_size`: 学習ステップに渡すバッチサイズ。

**実行例:**

```bash
# AIワークフローパイプライン全体を実行
python yggdrasil.py pipeline_orchestrator \
    --set epochs=3 \
    --set batch_size=32
```

このコマンドは、以下のステップを順番に実行します。

1.  `data_preprocessor.py` を使用してデータを生成・前処理。
2.  `mnist_trainer.py` を使用してモデルを学習。
3.  `model_evaluator.py` を使用してモデルを評価。

これらのステップは、MLflowの親ランと子ランとして自動的に追跡されます。

## MLflow連携

Yggdrasil Agent Framework は、MLflowと密接に連携し、AI実験の追跡と管理を容易にします。

### MLflow UIの起動方法

実験結果を視覚的に確認するには、MLflow UIを起動します。プロジェクトのルートディレクトリで、仮想環境をアクティベートしてから以下のコマンドを実行します。

```bash
source .venv/bin/activate
mlflow ui
```

通常、MLflow UIは `http://localhost:5000` で利用可能になります。ウェブブラウザでこのURLにアクセスしてください。

### 実験結果の確認方法

MLflow UIでは、以下の情報を確認できます。

*   **実験一覧:** 実行されたすべての実験（ラン）が一覧表示されます。
*   **ランの詳細:** 各ランをクリックすると、そのランで記録されたパラメータ、メトリクス、アーティファクト（モデルファイルなど）の詳細が表示されます。
*   **ネストされたラン:** `pipeline_orchestrator` のようなパイプライン実行では、親ランの下に各ステップの子ランがネストされて表示されます。これにより、パイプライン全体のパフォーマンスと各ステップの貢献度を一度に確認できます。
*   **比較:** 複数のランを選択して比較することで、異なるパラメータ設定やモデルのパフォーマンスの違いを分析できます。

## プロジェクト構造

Yggdrasil Agent Framework の主要なディレクトリとファイルの構造は以下の通りです。

```
yggdrasil-agent-framework/
├── .venv/                  # 仮想環境ディレクトリ
├── agents/                 # エージェント定義ファイル
│   ├── hello_agent.py
│   ├── model_trainer.py
│   └── pipeline_orchestrator.py
├── data/                   # データファイル (生成されたデータなど)
├── logs/                   # ログファイル (MLflowログ、CSVログなど)
├── trained_models/         # 学習済みモデルの保存先
├── training_scripts/       # 学習およびデータ処理スクリプト
│   ├── data_preprocessor.py
│   ├── mnist_trainer.py
│   └── model_evaluator.py
├── yggdrasil.py            # フレームワークのメインディスパッチャ
├── requirements.txt        # Pythonの依存関係リスト
├── GEMINI_RESEARCH_LOG.txt # 開発ログ
└── USER_GUIDE.md           # このユーザーガイド
```

*   `yggdrasil.py`: フレームワークのエントリポイント。エージェントのロードと実行を管理します。
*   `agents/`: 各タスクを実行するエージェントのPythonスクリプトが配置されます。
*   `training_scripts/`: エージェントによって呼び出される実際の学習、データ処理、評価のロジックが含まれるスクリプトです。
*   `.venv/`: 仮想環境が作成される場所です。
*   `data/`, `logs/`, `trained_models/`: 生成されたデータ、ログ、学習済みモデルが保存されるデフォルトの場所です。
*   `requirements.txt`: プロジェクトの依存関係を定義します。
*   `GEMINI_RESEARCH_LOG.txt`: 開発の進捗と決定事項を記録したログです。
*   `USER_GUIDE.md`: このユーザーガイドです。

## フレームワークの拡張

Yggdrasil Agent Framework は、新しい機能やワークフローを簡単に追加できるように設計されています。

### 新しいエージェントの作成方法

新しいエージェントを作成するには、以下の手順に従います。

1.  `agents/` ディレクトリ内に新しいPythonファイル（例: `my_new_agent.py`）を作成します。
2.  ファイル内に `main(args, config)` 関数を定義します。この関数がエージェントのエントリポイントとなります。
3.  `main` 関数内で、エージェントが実行するロジックを記述します。必要に応じて、`config` 辞書からパラメータを取得したり、`subprocess` モジュールを使用して他のスクリプトやエージェントを呼び出したりできます。
4.  新しいエージェントは、`python yggdrasil.py my_new_agent` のように実行できます。

### 新しい学習スクリプトの追加方法

`model_trainer` エージェントで使用する新しい学習スクリプトを追加するには、以下の手順に従います。

1.  `training_scripts/` ディレクトリ内に新しいPythonファイル（例: `my_custom_trainer.py`）を作成します。
2.  スクリプト内で `argparse` を使用してコマンドライン引数をパースし、必要なパラメータを受け取れるようにします。
3.  スクリプトのメインロジックを記述します。
4.  `model_trainer` エージェントからこのスクリプトを呼び出すには、`--agent-set script_path=training_scripts/my_custom_trainer.py` のように指定します。

## トラブルシューティング

### `mlflow` コマンドが見つからない

これは、MLflowが仮想環境にインストールされているが、仮想環境がアクティベートされていない場合に発生します。エージェントを実行する前、または `mlflow ui` を実行する前に、必ず仮想環境をアクティベートしてください。

```bash
source .venv/bin/activate
```

### エージェントの実行時に `unrecognized arguments` エラーが発生する

これは、エージェントに渡された引数が、そのエージェントまたはそのエージェントが呼び出すスクリプトで認識されない場合に発生します。以下の点を確認してください。

*   引数名が正しいか（例: `--epochs` ではなく `--epoch` となっていないか）。
*   引数が、そのスクリプトで `argparse` などによって定義されているか。
*   `pipeline_orchestrator` のように、複数のエージェントを呼び出すエージェントの場合、子エージェントに渡される引数が正しくフィルタリングされているか（例: `data_preprocessor.py` に `parent_run_id` が渡されていないか）。

### MLflow UIで実験が表示されない、または正しくネストされない

以下の点を確認してください。

*   MLflow UIが正しく起動しているか (`http://localhost:5000` にアクセスできるか)。
*   `mlflow.start_run()` が正しく呼び出されているか。
*   ネストされたランの場合、子ランが `mlflow.start_run(nested=True)` で開始されているか。
*   親ランの `run_id` が子ランに正しく渡されているか。

### その他の問題

上記で解決しない場合は、以下の情報を確認してください。

*   ターミナルに表示されるエラーメッセージの全文。
*   `logs/` ディレクトリ内のログファイル。
*   `GEMINI_RESEARCH_LOG.txt` を参照し、過去の開発で同様の問題が発生していないか確認してください。
