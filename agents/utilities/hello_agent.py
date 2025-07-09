#!/usr/bin/env python3
# DESCRIPTION: Hello World Agent

DEFAULT_CONFIG = {
    "greeting": "Hello",
    "target": "World",
    "message": "Default message from Hello Agent!"
}

def main(args, config=None):
    # デフォルト設定と渡された設定をマージ
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)

    print(f"{final_config['greeting']} from Hello Agent! (Target: {final_config['target']})")
    if args:
        print(f"Received arguments: {args}")
    if final_config:
        print(f"Final configuration: {final_config}")
        if "message" in final_config:
            print(f"Configured message: {final_config['message']}")

if __name__ == "__main__":
    main([], {}) # デフォルトで空の引数と設定を渡す
