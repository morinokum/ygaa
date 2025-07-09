import os
import json
import argparse

# プロジェクトルートを定義 (このスクリプトがどこにあっても動作するように調整)
# このスクリプトが my_yggdrasil_framework/agents/ の直下にあると仮定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FILE = os.path.join(PROJECT_ROOT, "data", "models.json")

def load_models():
    if not os.path.exists(MODELS_FILE):
        return []
    try:
        with open(MODELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {MODELS_FILE}. Returning empty list.", file=os.sys.stderr)
        return []

def save_models(models):
    os.makedirs(os.path.dirname(MODELS_FILE), exist_ok=True)
    with open(MODELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(models, f, ensure_ascii=False, indent=4)

def add_model(name, comment):
    models = load_models()
    models.append({"name": name, "comment": comment})
    save_models(models)
    print(f"Model '{name}' added to collection.")

def list_models():
    models = load_models()
    if not models:
        print("No models in the collection yet.")
        return

    print("\n--- Model Collection ---")
    for i, model in enumerate(models):
        print(f"{i + 1}. Name: {model['name']}")
        print(f"   Comment: {model['comment']}")
        print("-" * 20)
    print("--------------------")

def update_model(name, comment):
    models = load_models()
    found = False
    for model in models:
        if model["name"] == name:
            model["comment"] = comment
            found = True
            break
    if found:
        save_models(models)
        print(f"Model '{name}' comment updated successfully.")
    else:
        print(f"Error: Model '{name}' not found.", file=os.sys.stderr)

def main(args, config):
    parser = argparse.ArgumentParser(description='Learning Model Collection Tool')
    parser.add_argument('--add', action='store_true', help='Add a new model to the collection.')
    parser.add_argument('--list', action='store_true', help='List models in the collection.')
    parser.add_argument('--update', action='store_true', help="Update an existing model's comment.")
    parser.add_argument('--name', type=str, help='Name of the model.')
    parser.add_argument('--comment', type=str, help='Comment about the model.')

    # yggdrasil.pyから渡される引数をパース
    parsed_args = parser.parse_args(args)

    if parsed_args.add:
        if not parsed_args.name or not parsed_args.comment:
            print("Error: Name and comment are required to add a model.", file=os.sys.stderr)
        else:
            add_model(parsed_args.name, parsed_args.comment)
    elif parsed_args.list:
        list_models()
    elif parsed_args.update:
        if not parsed_args.name or not parsed_args.comment:
            print("Error: Model name and new comment are required to update a model.", file=os.sys.stderr)
        else:
            update_model(parsed_args.name, parsed_args.comment)
    else:
        print("Please specify a command: --add, --list, or --update", file=os.sys.stderr)

if __name__ == '__main__':
    # このスクリプトが直接実行された場合のテスト用
    # yggdrasil.pyから呼び出される場合は、main関数に引数が渡される
    main(os.sys.argv[1:], {}) # コマンドライン引数を渡し、空のconfigを渡す