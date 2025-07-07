import os
import subprocess
import sys

# プロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_git_command(command_parts, cwd=PROJECT_ROOT, check=False):
    """Gitコマンドを実行し、stdoutとstderrを返す"""
    try:
        result = subprocess.run(
            command_parts,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check # Trueの場合、非ゼロ終了コードでCalledProcessErrorを発生
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        print(f"エラー: Gitコマンドが見つかりません。Gitがインストールされ、PATHが通っているか確認してください。", file=sys.stderr)
        return "", "Git not found", 1
    except subprocess.CalledProcessError as e:
        print(f"エラー: Gitコマンドの実行に失敗しました。リターンコード: {e.returncode}\n{e.stderr}", file=sys.stderr)
        return e.stdout, e.stderr, e.returncode

def main():
    print("\n--- GitHubプッシュヘルパーエージェント ---\n")
    print("このエージェントは、あなたのプロジェクトをGitHubにプッシュする手順をガイドします。")
    print("セキュリティ上の理由により、Gitの認証情報（パスワードなど）を直接扱うことはできません。")
    print("そのため、最終的なGitコマンドはあなた自身でコピーして実行していただく必要があります。\n")

    # 1. Gitリポジトリの状態を確認
    print("## 1. 現在のGitリポジトリの状態を確認します...\n")
    stdout, stderr, returncode = run_git_command(["git", "status"])
    print(stdout)
    if "fatal: not a git repository" in stderr:
        print("\nエラー: 現在のディレクトリはGitリポジリではありません。")
        print(f"プロジェクトのルートディレクトリ ({PROJECT_ROOT}) でGitが初期化されているか確認してください。")
        print("もし初期化されていない場合は、以下のコマンドを実行してください:")
        print(f"    `cd {PROJECT_ROOT}`")
        print(f"    `git init`")
        print("その後、再度このヘルパーエージェントを実行してください。")
        return
    elif returncode != 0:
        print(f"Git statusの実行中に予期せぬエラーが発生しました:\n{stderr}", file=sys.stderr)
        return

    # 2. リモートリポジトリの設定を確認
    print("\n## 2. リモートリポジトリの設定を確認します...\n")
    stdout_remote, stderr_remote, returncode_remote = run_git_command(["git", "remote", "-v"])
    print(stdout_remote)

    remote_configured = "origin" in stdout_remote

    if not remote_configured:
        print("\n### リモートリポジトリ 'origin' が設定されていません。")
        print("GitHubにプロジェクトをプッシュするには、まずリモートリポジトリを設定する必要があります。")
        print("以下の手順でGitHubリポジトリを作成し、そのURLをここに入力してください。\n")
        print("**手順:**")
        print("1.  ウェブブラウザでGitHubにアクセスし、ログインします。")
        print("2.  新しいリポジトリを作成します (例: 'my_yggdrasil_framework')。")
        print("3.  リポジトリ作成後、表示される '...or push an existing repository from the command line' のセクションにある 'git remote add origin ...' のコマンドをコピーします。")
        print("    例: `git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git`\n")
        print("**コピーしたコマンドをここに貼り付けてください。**")
        print("    (例: `git remote add origin https://github.com/your_username/my_yggdrasil_framework.git`)")
        
        remote_add_command = input("コマンドを貼り付けてEnter: ")
        if remote_add_command.startswith("git remote add origin"): # ユーザー入力の検証
            command_parts = remote_add_command.split()
            print(f"\n--- コマンド実行中: {' '.join(command_parts)} ---")
            stdout_add, stderr_add, returncode_add = run_git_command(command_parts, check=True)
            print(stdout_add)
            if returncode_add == 0:
                print("リモートリポジトリ 'origin' が正常に設定されました。")
                remote_configured = True
            else:
                print(f"リモートリポジトリの設定中にエラーが発生しました:\n{stderr_add}", file=sys.stderr)
                print("設定を修正して、再度このヘルパーエージェントを実行してください。")
                return
        else:
            print("無効なコマンド形式です。'git remote add origin ...' の形式で入力してください。")
            return
    else:
        print("リモートリポジトリ 'origin' は既に設定されています。")

    # 3. プッシュの準備
    if remote_configured:
        print("\n## 3. GitHubへのプッシュ準備ができました。")
        print("以下のコマンドをコピーして、あなたのターミナルで実行してください。")
        print("これにより、ローカルの変更がGitHubリポジトリにアップロードされます。\n")
        print("**プッシュコマンド:**")
        
        # 現在のブランチ名を取得
        stdout_branch, _, _ = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        current_branch = stdout_branch.strip()
        
        push_command = f"git push -u origin {current_branch}"
        print(f"```bash\n{push_command}\n```")
        
        print("\n**注意点:**")
        print("1.  このコマンドを実行すると、GitHubのユーザー名とパスワード、またはパーソナルアクセストークンの入力を求められる場合があります。")
        print("2.  パスワード入力時には、画面に文字が表示されませんが、入力は受け付けられています。入力後にEnterを押してください。")
        print("3.  もしエラーが発生した場合は、表示されるメッセージをよく読み、必要に応じてGitHubのドキュメントを参照してください。")
        print("4.  初回プッシュ後は、`-u origin {branch_name}` の部分は省略して `git push` だけでプッシュできるようになります。")
        print("\n--- ガイドは以上です。頑張ってください！ ---\n")

if __name__ == "__main__":
    main()
