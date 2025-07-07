import pytest
import os
import sys
from unittest.mock import patch

# エージェントモジュールをインポートするためにsys.pathに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents')))
import hello_agent

# config_utilsをインポート
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from config_utils import merge_configs

def test_hello_agent_no_args(capsys):
    """
    引数なしでhello_agentを実行した場合の出力を確認
    """
    hello_agent.main([], {}) # 引数と設定を渡すように変更
    captured = capsys.readouterr()
    expected_final_config = hello_agent.DEFAULT_CONFIG.copy()
    assert f"{expected_final_config['greeting']} from Hello Agent! (Target: {expected_final_config['target']})\n" in captured.out
    assert "Received arguments:" not in captured.out
    assert f"Final configuration: {expected_final_config}\n" in captured.out
    assert f"Configured message: {expected_final_config['message']}\n" in captured.out

def test_hello_agent_with_args(capsys):
    """
    引数ありでhello_agentを実行した場合の出力を確認
    """
    test_args = ["arg1", "arg2"]
    hello_agent.main(test_args, {}) # 引数と設定を渡すように変更
    captured = capsys.readouterr()
    expected_final_config = hello_agent.DEFAULT_CONFIG.copy()
    assert f"{expected_final_config['greeting']} from Hello Agent! (Target: {expected_final_config['target']})\n" in captured.out
    assert f"Received arguments: {test_args}\n" in captured.out
    assert f"Final configuration: {expected_final_config}\n" in captured.out
    assert f"Configured message: {expected_final_config['message']}\n" in captured.out

def test_hello_agent_with_config(capsys):
    """
    設定ありでhello_agentを実行した場合の出力を確認
    """
    test_config = {"message": "Test Message", "version": "2.0", "target": "User"}
    hello_agent.main([], test_config)
    captured = capsys.readouterr()
    expected_final_config = merge_configs(hello_agent.DEFAULT_CONFIG, test_config)
    assert f"{expected_final_config['greeting']} from Hello Agent! (Target: {expected_final_config['target']})\n" in captured.out
    assert "Received arguments:" not in captured.out
    assert f"Final configuration: {expected_final_config}\n" in captured.out
    assert f"Configured message: {expected_final_config['message']}\n" in captured.out

def test_hello_agent_with_args_and_config(capsys):
    """
    引数と設定ありでhello_agentを実行した場合の出力を確認
    """
    test_args = ["arg1", "arg2"]
    test_config = {"message": "Another Test Message", "greeting": "Hi"}
    hello_agent.main(test_args, test_config)
    captured = capsys.readouterr()
    expected_final_config = merge_configs(hello_agent.DEFAULT_CONFIG, test_config)
    assert f"{expected_final_config['greeting']} from Hello Agent! (Target: {expected_final_config['target']})\n" in captured.out
    assert f"Received arguments: {test_args}\n" in captured.out
    assert f"Final configuration: {expected_final_config}\n" in captured.out
    assert f"Configured message: {expected_final_config['message']}\n" in captured.out
