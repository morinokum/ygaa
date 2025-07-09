#!/usr/bin/env python3
# DESCRIPTION: Agent for managing other agents (list, create)

import os
import sys
from utils import logger

# プロジェクトルートとエージェントディレクトリを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")

def list_agents():
    logger.info("\n--- 利用可能なエージェント一覧 ---")
    found_agents = []
    for f in os.listdir(AGENTS_DIR):
        if f.endswith(".py") and f != "__init__.py" and f != os.path.basename(__file__):
            found_agents.append(f[:-3])
    
    if found_agents:
        for agent_name in sorted(found_agents):
            logger.info(f"- {agent_name}")
    else:
        logger.info("エージェントが見つかりません。")
    logger.info("----------------------------------")

def create_agent(agent_name):
    if not agent_name:
        logger.error("エラー: 作成するエージェントの名前を指定してください。")
        return

    new_agent_path = os.path.join(AGENTS_DIR, f"{agent_name}.py")
    if os.path.exists(new_agent_path):
        logger.error(f"エラー: エージェント '{agent_name}' は既に存在します。")
        return

    template_content = f"""#!/usr/bin/env python3
# DESCRIPTION: {agent_name} Agent

def main(args):
    print("Hello from {agent_name} Agent!")
    if args:
        print(f"Received arguments: {{args}}")

if __name__ == "__main__":
    main([])
"""

    try:
        with open(new_agent_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        os.chmod(new_agent_path, 0o755) # 実行権限を付与
        logger.info(f"エージェント '{agent_name}' を作成しました: {new_agent_path}")
    except Exception as e:
        logger.error(f"エラー: エージェント '{agent_name}' の作成中にエラーが発生しました: {e}")

def main(args):
    if not args:
        logger.info("使用方法: manage_agents <command> [args...]")
        logger.info("コマンド:")
        logger.info("  list        - 利用可能なエージェントを一覧表示します。")
        logger.info("  create <name> - 新しいエージェントのひな形を作成します。")
        return

    command = args[0]
    if command == "list":
        list_agents()
    elif command == "create":
        if len(args) > 1:
            create_agent(args[1])
        else:
            logger.error("エラー: 'create' コマンドにはエージェントの名前が必要です。")
    else:
        logger.error(f"エラー: 不明なコマンド '{command}' です。")

if __name__ == "__main__":
    # 単体テスト用
    # main(["list"])
    # main(["create", "my_new_agent"])
    main(sys.argv[1:])
