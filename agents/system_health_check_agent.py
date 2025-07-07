#!/usr/bin/env python3
# DESCRIPTION: Agent for performing basic system health checks.

import os
import shutil
from utils import logger

# プロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

def check_directory_exists(path, description):
    if os.path.exists(path):
        logger.info(f"[OK] {description} ディレクトリが見つかりました: {path}")
        return True
    else:
        logger.error(f"[NG] {description} ディレクトリが見つかりません: {path}")
        return False

def check_disk_space(path, threshold_gb=1.0):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    if free_gb < threshold_gb:
        logger.warning(f"[NG] ディスク空き容量が少ないです ({free_gb:.2f} GB)。閾値: {threshold_gb} GB")
        return False
    else:
        logger.info(f"[OK] ディスク空き容量: {free_gb:.2f} GB")
        return True

def check_log_file_size(log_file_path, threshold_mb=10.0):
    if os.path.exists(log_file_path):
        file_size_mb = os.path.getsize(log_file_path) / (1024**2)
        if file_size_mb > threshold_mb:
            logger.warning(f"[NG] ログファイルが大きすぎます ({file_size_mb:.2f} MB)。閾値: {threshold_mb} MB")
            return False
        else:
            logger.info(f"[OK] ログファイルサイズ: {file_size_mb:.2f} MB")
            return True
    else:
        logger.info(f"[OK] ログファイルが見つかりません: {log_file_path}")
        return True

def main(args):
    logger.info("--- システム健全性チェックを開始します ---")
    overall_status = True

    # 1. 主要ディレクトリの存在チェック
    logger.info("\n--- ディレクトリ存在チェック ---")
    if not check_directory_exists(CONFIG_DIR, "設定"): overall_status = False
    if not check_directory_exists(AGENTS_DIR, "エージェント"): overall_status = False
    if not check_directory_exists(LOGS_DIR, "ログ"): overall_status = False
    if not check_directory_exists(UTILS_DIR, "ユーティリティ"): overall_status = False

    # 2. ディスク空き容量チェック (プロジェクトルート)
    logger.info("\n--- ディスク空き容量チェック ---")
    if not check_disk_space(PROJECT_ROOT, threshold_gb=0.5): overall_status = False # 0.5GB を閾値とする

    # 3. メインログファイルのサイズチェック
    logger.info("\n--- ログファイルサイズチェック ---")
    main_log_file = os.path.join(LOGS_DIR, "yggdrasil.log")
    if not check_log_file_size(main_log_file, threshold_mb=5.0): overall_status = False # 5MB を閾値とする

    logger.info("\n--- システム健全性チェックが完了しました ---")
    if overall_status:
        logger.info("全体的なシステムの状態は良好です。")
    else:
        logger.warning("システムにいくつかの問題が検出されました。ログを確認してください。")

if __name__ == "__main__":
    main([])
