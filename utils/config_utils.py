def merge_configs(base_config, *override_configs):
    """
    複数の設定辞書をマージします。後の辞書が前の辞書を上書きします。
    ネストされた辞書も再帰的にマージします。
    """
    merged_config = base_config.copy()
    for override_config in override_configs:
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key] = merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
    return merged_config
