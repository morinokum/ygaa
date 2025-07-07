#!/usr/bin/env python3
# DESCRIPTION: Yggdrasil Agent Framework - Main Dispatcher

import argparse
import os
import sys
import json
from utils import logger
from utils.config_utils import merge_configs

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
AGENT_CONFIG_DIR = os.path.join(AGENTS_DIR, "config")  # エージェント固有の設定ディレクトリ
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

# デフォルトのフレームワーク設定
DEFAULT_FRAMEWORK_CONFIG = {
    "logging": {
        "level": "INFO",
        "file": "yggdrasil.log"
    },
    "agent_execution": {
        "timeout_seconds": 60
    }
}

# 設定ファイルをロードする関数
def load_config():
    config_path = os.path.join(CONFIG_DIR, "framework_config.json")
    if not os.path.exists(config_path):
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"設定ファイルの解析に失敗しました: {config_path}")
        return {}

# エージェント固有の設定ファイルをロードする関数
def load_agent_config(agent_name):
    config_path = os.path.join(AGENT_CONFIG_DIR, f"{agent_name}.json")
    if not os.path.exists(config_path):
        logger.info(f"エージェント '{agent_name}' の設定ファイルが見つかりません: {config_path} (スキップします)")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"エージェント '{agent_name}' の設定ファイルの解析に失敗しました: {config_path}")
        return {}

# コマンドライン引数から設定をパースするヘルパー関数
def parse_set_args(set_args):
    parsed_config = {}
    for arg in set_args:
        if '=' not in arg:
            logger.warning(f"無効な --set 引数形式: {arg}。KEY=VALUE の形式を期待します。")
            continue
        key, value = arg.split('=', 1)
        keys = key.split('.')
        current_dict = parsed_config
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                if value.lower() == 'true':
                    current_dict[k] = True
                elif value.lower() == 'false':
                    current_dict[k] = False
                elif value.isdigit():
                    current_dict[k] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    current_dict[k] = float(value)
                else:
                    current_dict[k] = value
            else:
                if k not in current_dict:
                    current_dict[k] = {}
                elif not isinstance(current_dict[k], dict):
                    logger.warning(f"設定できません {key}: {k} は辞書ではありません。")
                    break
                current_dict = current_dict[k]
    return parsed_config

# エージェントを実行する関数
def run_agent(agent_name, args, framework_config):
    agent_path = os.path.join(AGENTS_DIR, f"{agent_name}.py")
    if not os.path.exists(agent_path):
        logger.error(f"エージェント '{agent_name}' が見つかりません。")
        return

    agent_config_from_file = load_agent_config(agent_name)
    sys.path.insert(0, AGENTS_DIR)

    try:
        agent_module = __import__(agent_name)
        default_agent_config = getattr(agent_module, 'DEFAULT_CONFIG', {})

        agent_parser = argparse.ArgumentParser(add_help=False)
        agent_parser.add_argument('--agent-set', action='append', default=[], help='エージェント固有の設定を KEY=VALUE 形式で上書き')
        parsed_agent_args, remaining_agent_args = agent_parser.parse_known_args(args)
        agent_config_from_cli = parse_set_args(parsed_agent_args.agent_set)

        final_agent_config = merge_configs(default_agent_config, agent_config_from_file, agent_config_from_cli)

        logger.info(f"--- エージェント '{agent_name}' を実行中 ---")
        if hasattr(agent_module, 'main') and callable(agent_module.main):
            agent_module.main(remaining_agent_args, final_agent_config)
        else:
            logger.warning(f"警告: エージェント '{agent_name}' に 'main' 関数が見つからないか、呼び出し可能ではありません。")
    except ImportError:
        logger.error(f"エラー: エージェント '{agent_name}' のインポートに失敗しました。")
    except Exception as e:
        logger.error(f"エラー: エージェント '{agent_name}' の実行中に例外が発生しました: {str(e)}")
    finally:
        if AGENTS_DIR in sys.path:
            sys.path.remove(AGENTS_DIR)

    logger.info(f"--- エージェント '{agent_name}' の実行が完了しました ---")

def main():
    framework_parser = argparse.ArgumentParser(add_help=False)
    framework_parser.add_argument('--set', action='append', default=[], help='フレームワーク設定を KEY=VALUE 形式で上書き')
    framework_parser.add_argument("agent", nargs="?", help="実行するエージェントの名前")

    known_args, agent_args = framework_parser.parse_known_args()
    framework_config_from_file = load_config()
    framework_config_from_cli = parse_set_args(known_args.set)
    final_framework_config = merge_configs(DEFAULT_FRAMEWORK_CONFIG, framework_config_from_file, framework_config_from_cli)

    if known_args.agent:
        run_agent(known_args.agent, agent_args, final_framework_config)
    else:
        logger.info("Yggdrasil Agent Framework")
        logger.info("使用方法: python3 yggdrasil.py <agent_name> [agent_args...] [--set KEY=VALUE] [--agent-set KEY=VALUE]")
        logger.info("")
        logger.info("利用可能なエージェント:")
        for f in os.listdir(AGENTS_DIR):
            if f.endswith(".py") and f != "__init__.py":
                logger.info(f"- {f[:-3]}")

if __name__ == "__main__":
    main()
