import os
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock
import sys
import importlib.util

# yggdrasil.py から load_config 関数と run_agent 関数、関連定数をインポート
# テスト対象のモジュールをsys.pathに追加する必要がある
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yggdrasil import load_config, run_agent, load_agent_config, parse_set_args, PROJECT_ROOT, CONFIG_DIR, AGENTS_DIR, AGENT_CONFIG_DIR, DEFAULT_FRAMEWORK_CONFIG
from utils.config_utils import merge_configs

@pytest.fixture
def mock_logger():
    with patch('yggdrasil.logger') as mock_log:
        yield mock_log

# --- load_config 関数のテスト (既存) ---
def test_load_config_file_not_found(mock_logger):
    """
    設定ファイルが見つからない場合に空の辞書を返し、エラーをログに記録することを確認
    """
    with patch('os.path.exists', return_value=False):
        config = load_config()
        assert config == {}
        mock_logger.error.assert_called_with(f"設定ファイルが見つかりません: {os.path.join(CONFIG_DIR, "framework_config.json")}")

def test_load_config_json_decode_error(mock_logger):
    """
    設定ファイルのJSON解析に失敗した場合に空の辞書を返し、エラーをログに記録することを確認
    """
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='invalid json')):
            config = load_config()
            assert config == {}
            mock_logger.error.assert_called_with(f"設定ファイルの解析に失敗しました: {os.path.join(CONFIG_DIR, "framework_config.json")}")

def test_load_config_success(mock_logger):
    """
    設定ファイルが正常にロードされた場合に正しい辞書を返すことを確認
    """
    mock_config_data = {"key": "value", "number": 123}
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_config_data))):
            config = load_config()
            assert config == mock_config_data
            mock_logger.error.assert_not_called() # エラーがログに記録されないことを確認

# --- load_agent_config 関数のテスト (既存) ---
def test_load_agent_config_file_not_found(mock_logger):
    """
    エージェント設定ファイルが見つからない場合に空の辞書を返し、情報をログに記録することを確認
    """
    with patch('os.path.exists', return_value=False):
        config = load_agent_config("test_agent")
        assert config == {}
        mock_logger.info.assert_called_with(f"エージェント 'test_agent' の設定ファイルが見つかりません: {os.path.join(AGENT_CONFIG_DIR, "test_agent.json")} (スキップします)")

def test_load_agent_config_json_decode_error(mock_logger):
    """
    エージェント設定ファイルのJSON解析に失敗した場合に空の辞書を返し、エラーをログに記録することを確認
    """
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='invalid json')):
            config = load_agent_config("test_agent")
            assert config == {}
            mock_logger.error.assert_called_with(f"エージェント 'test_agent' の設定ファイルの解析に失敗しました: {os.path.join(AGENT_CONFIG_DIR, "test_agent.json")}")

def test_load_agent_config_success(mock_logger):
    """
    エージェント設定ファイルが正常にロードされた場合に正しい辞書を返すことを確認
    """
    mock_config_data = {"agent_key": "agent_value"}
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_config_data))):
            config = load_agent_config("test_agent")
            assert config == mock_config_data
            mock_logger.error.assert_not_called() # エラーがログに記録されないことを確認

# --- parse_set_args 関数のテスト (新規) ---
def test_parse_set_args_simple():
    args = ["key1=value1", "key2=123", "key3=true", "key4=3.14"]
    expected = {"key1": "value1", "key2": 123, "key3": True, "key4": 3.14}
    assert parse_set_args(args) == expected

def test_parse_set_args_nested():
    args = ["nested.key1=value1", "nested.sub.key2=false"]
    expected = {"nested": {"key1": "value1", "sub": {"key2": False}}}
    assert parse_set_args(args) == expected

def test_parse_set_args_invalid_format(mock_logger):
    args = ["invalid_format", "key=value"]
    expected = {"key": "value"}
    result = parse_set_args(args)
    assert result == expected
    mock_logger.warning.assert_called_with("無効な --set 引数形式: invalid_format。KEY=VALUE の形式を期待します。")

def test_parse_set_args_non_dict_override(mock_logger):
    args = ["a=1", "a.b=2"]
    expected = {"a": 1}
    result = parse_set_args(args)
    assert result == expected
    mock_logger.warning.assert_called_with("設定できません a.b: a は辞書ではありません。")

# --- run_agent 関数のテスト (既存 + 修正) ---

@pytest.fixture
def mock_agent_module():
    """ダミーのエージェントモジュールを作成するフィクスチャ"""
    mock_module = MagicMock()
    mock_module.__name__ = "test_agent"
    mock_module.__file__ = os.path.join(AGENTS_DIR, "test_agent.py")
    # エージェントのデフォルト設定をモックモジュールに追加
    mock_module.DEFAULT_CONFIG = {"default_agent_key": "default_agent_value"}
    # Directly add to sys.modules for robust mocking
    sys.modules['test_agent'] = mock_module
    yield mock_module
    # Clean up sys.modules after test
    if 'test_agent' in sys.modules:
        del sys.modules['test_agent']

@pytest.fixture
def mock_import_module():
    with patch('importlib.util.spec_from_file_location') as mock_spec_from_file_location,\
         patch('importlib.util.module_from_spec') as mock_module_from_spec:
        mock_spec = MagicMock()
        mock_spec_from_file_location.return_value = mock_spec
        mock_agent_module = MagicMock()
        mock_agent_module.__name__ = "test_agent"
        mock_agent_module.__file__ = os.path.join(AGENTS_DIR, "test_agent.py")
        mock_agent_module.DEFAULT_CONFIG = {"default_agent_key": "default_agent_value"}
        mock_module_from_spec.return_value = mock_agent_module
        yield mock_agent_module


def test_run_agent_not_found(mock_logger):
    """
    エージェントファイルが見つからない場合にエラーをログに記録することを確認
    """
    with patch('os.path.exists', return_value=False):
        framework_config = {"framework_key": "framework_value"} # framework_config を定義
        run_agent("non_existent_agent", [], framework_config) # framework_config を追加
        mock_logger.error.assert_called_with("エージェント 'non_existent_agent' が見つかりません。")

def test_run_agent_import_error(mock_logger, mock_import_module):
    """
    エージェントのインポートに失敗した場合にエラーをログに記録することを確認
    """
    with patch('os.path.exists', return_value=True):
        with patch('yggdrasil.load_agent_config', return_value={}): # load_agent_configをモック化
            mock_import_module.side_effect = ImportError("Mock Import Error")
            framework_config = {"framework_key": "framework_value"}
            run_agent("test_agent", [], framework_config) # framework_config を追加
            mock_logger.error.assert_called_with("エラー: エージェント 'test_agent' のインポートに失敗しました。")

def test_run_agent_no_main_function(mock_logger, mock_agent_module):
    """
    エージェントにmain関数がない場合に警告をログに記録することを確認
    """
    del mock_agent_module.main # main関数を削除
    with patch('os.path.exists', return_value=True):
        with patch('yggdrasil.load_agent_config', return_value={}): # load_agent_configをモック化
            framework_config = {"framework_key": "framework_value"}
            framework_config = {"framework_key": "framework_value"}
            run_agent("test_agent", [], framework_config) # framework_config を追加
            mock_logger.warning.assert_called_with("警告: エージェント 'test_agent' に 'main' 関数が見つからないか、呼び出し可能ではありません。")
            assert not hasattr(mock_agent_module, 'main') # main関数が存在しないことを確認

def test_run_agent_main_not_callable(mock_logger, mock_agent_module):
    """
    エージェントのmain関数が呼び出し可能でない場合に警告をログに記録することを確認
    """
    mock_agent_module.main = "not_callable" # main関数を呼び出し不可にする
    with patch('os.path.exists', return_value=True):
        with patch('yggdrasil.load_agent_config', return_value={}): # load_agent_configをモック化
            framework_config = {"framework_key": "framework_value"}
            framework_config = {"framework_key": "framework_value"}
            run_agent("test_agent", [], framework_config) # framework_config を追加
            mock_logger.warning.assert_called_with("警告: エージェント 'test_agent' に 'main' 関数が見つからないか、呼び出し可能ではありません。")

def test_run_agent_success(mock_logger, mock_agent_module):
    """
    エージェントが正常に実行された場合にmain関数が呼び出されることを確認
    """
    mock_agent_module.main = MagicMock() # main関数をモック化
    test_args = ["arg1", "arg2"]
    mock_agent_config_from_file = {"file_key": "file_value"}
    expected_final_config = merge_configs(mock_agent_module.DEFAULT_CONFIG, mock_agent_config_from_file)

    with patch('os.path.exists', return_value=True):
        with patch('yggdrasil.load_agent_config', return_value=mock_agent_config_from_file): # load_agent_configをモック化
            framework_config = {"framework_key": "framework_value"} # framework_config を定義
            run_agent("test_agent", test_args, framework_config) # framework_config を追加
            mock_agent_module.main.assert_called_once_with(test_args, expected_final_config)
            mock_logger.info.assert_any_call("--- エージェント 'test_agent' を実行中 ---")
            mock_logger.info.assert_any_call("--- エージェント 'test_agent' の実行が完了しました ---")
            mock_logger.error.assert_not_called() # エラーがログに記録されないことを確認

def test_run_agent_exception_during_execution(mock_logger, mock_agent_module):
    """
    エージェントの実行中に例外が発生した場合にエラーをログに記録することを確認
    """
    mock_agent_module.main = MagicMock(side_effect=ValueError("Mock Agent Error"))
    with patch('os.path.exists', return_value=True):
        with patch('yggdrasil.load_agent_config', return_value={}): # load_agent_configをモック化
            framework_config = {"framework_key": "framework_value"}
            framework_config = {"framework_key": "framework_value"}
            run_agent("test_agent", [], framework_config) # framework_config を追加
            mock_logger.error.assert_called_with("エラー: エージェント 'test_agent' の実行中に例外が発生しました: Mock Agent Error")

def test_run_agent_with_cli_agent_config_override(mock_logger, mock_agent_module):
    """
    --agent-set 引数でエージェント設定が正しく上書きされることを確認
    """
    mock_agent_module.main = MagicMock() # main関数をモック化
    test_args = ["--agent-set", "agent_key=cli_value", "--agent-set", "nested.param=123"]
    mock_agent_config_from_file = {"agent_key": "file_value", "nested": {"param": 456}}
    cli_override_config = {"agent_key": "cli_value", "nested": {"param": 123}}
    expected_final_config = merge_configs(mock_agent_module.DEFAULT_CONFIG, mock_agent_config_from_file, cli_override_config)

    with patch('os.path.exists', return_value=True):
        with patch('yggdrasil.load_agent_config', return_value=mock_agent_config_from_file):
            framework_config = {"framework_key": "framework_value"}
            run_agent("test_agent", [], framework_config)

# --- main 関数のテスト (新規) ---
@patch('yggdrasil.load_config', return_value={})
@patch('yggdrasil.run_agent')
@patch('yggdrasil.logger')
def test_main_with_cli_framework_config_override(mock_logger, mock_run_agent, mock_load_config):
    """
    --set 引数でフレームワーク設定が正しく上書きされることを確認
    """
    # sys.argv をモック化
    original_argv = sys.argv
    try:
        sys.argv = ["yggdrasil.py", "test_agent", "--set", "logging.level=DEBUG", "--set", "agent_execution.timeout_seconds=120"]
        from yggdrasil import main
        main()

        expected_framework_config = merge_configs(
            DEFAULT_FRAMEWORK_CONFIG,
            {"logging": {"level": "DEBUG"}, "agent_execution": {"timeout_seconds": 120}}
        )
        mock_run_agent.assert_called_once_with("test_agent", [], expected_framework_config)
        mock_logger.error.assert_not_called()
    finally:
        sys.argv = original_argv

@patch('yggdrasil.load_config', return_value={})
@patch('yggdrasil.run_agent')
@patch('yggdrasil.logger')
def test_main_no_agent_specified(mock_logger, mock_run_agent, mock_load_config):
    """
    エージェントが指定されない場合に情報メッセージが表示されることを確認
    """
    test_sys_argv = ["yggdrasil.py"]
    with patch.object(sys, 'argv', test_sys_argv):
        with patch('os.listdir', return_value=["hello_agent.py", "manage_agents.py"]):
            from yggdrasil import main # main関数を再インポート
            main()

            mock_logger.info.assert_any_call("Yggdrasil Agent Framework")
            mock_logger.info.assert_any_call("使用方法: python3 yggdrasil.py <agent_name> [agent_args...] [--set KEY=VALUE] [--agent-set KEY=VALUE]")
            mock_logger.info.assert_any_call("\n利用可能なエージェント:")
            mock_logger.info.assert_any_call("- hello_agent")
            mock_logger.info.assert_any_call("- manage_agents")
            mock_run_agent.assert_not_called()

# --- merge_configs 関数のテスト (既存) ---
def test_merge_configs_simple():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    expected = {"a": 1, "b": 3, "c": 4}
    assert merge_configs(base, override) == expected

def test_merge_configs_nested():
    base = {"a": 1, "b": {"x": 10, "y": 20}}
    override = {"b": {"y": 30, "z": 40}, "c": 5}
    expected = {"a": 1, "b": {"x": 10, "y": 30, "z": 40}, "c": 5}
    assert merge_configs(base, override) == expected

def test_merge_configs_multiple_overrides():
    base = {"a": 1}
    override1 = {"b": 2}
    override2 = {"a": 3, "c": 4}
    expected = {"a": 3, "b": 2, "c": 4}
    assert merge_configs(base, override1, override2) == expected

def test_merge_configs_empty_overrides():
    base = {"a": 1, "b": 2}
    expected = {"a": 1, "b": 2}
    assert merge_configs(base) == expected

def test_merge_configs_non_dict_override():
    base = {"a": 1, "b": {"x": 10}}
    override = {"b": "new_value"}
    expected = {"a": 1, "b": "new_value"}
    assert merge_configs(base, override) == expected