#!/usr/bin/env python3
# DESCRIPTION: Simple Logging Utility for Yggdrasil Framework

import os
import datetime

# プロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ログレベル定義
LOG_LEVELS = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4
}

# デフォルトのログレベル
CURRENT_LOG_LEVEL = LOG_LEVELS.get(os.environ.get("YGGDRASIL_LOG_LEVEL", "INFO").upper(), LOG_LEVELS["INFO"])

def _log(level, message):
    if LOG_LEVELS[level] < CURRENT_LOG_LEVEL:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"

    # 標準出力にも表示
    print(log_message)

    # ログファイルに書き込み
    log_file_path = os.path.join(LOGS_DIR, "yggdrasil.log")
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    except Exception as e:
        print(f"エラー: ログファイルへの書き込みに失敗しました: {e}")

def debug(message):
    _log("DEBUG", message)

def info(message):
    _log("INFO", message)

def warning(message):
    _log("WARNING", message)

def error(message):
    _log("ERROR", message)

def critical(message):
    _log("CRITICAL", message)

if __name__ == "__main__":
    # テスト
    info("これは情報メッセージです。")
    warning("これは警告メッセージです。")
    error("これはエラーメッセージです。")
    debug("これはデバッグメッセージです。(環境変数 YGGDRASIL_LOG_LEVEL=DEBUG で表示)")
    critical("これは致命的なエラーメッセージです。")
