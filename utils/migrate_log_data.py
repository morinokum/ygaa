import csv
import os

# プロジェクトルートを定義
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def migrate_log_data(input_file_path, output_file_path):
    print(f"--- ログデータの移行を開始します: {os.path.basename(input_file_path)} ---")
    
    migrated_data = []
    header = []
    
    try:
        with open(input_file_path, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            
            # learning_rate 列のインデックスを特定
            lr_index = -1
            if 'learning_rate' in header:
                lr_index = header.index('learning_rate')
            else:
                # ヘッダーに learning_rate がない場合、追加
                header.insert(header.index('batch_size') + 1, 'learning_rate')
                lr_index = header.index('learning_rate')
            
            migrated_data.append(header)

            for row in reader:
                row_dict = dict(zip(header[:len(row)], row))
                
                # learning_rate が存在しない、または空の場合にデフォルト値 (0.001) を設定
                if 'learning_rate' not in row_dict or not row_dict['learning_rate']:
                    # epochsとbatch_sizeが数値の場合のみ、MNISTのログと判断して0.001を適用
                    # pipeline_orchestratorのログはepochsやbatch_sizeがファイルパスになっているため除外
                    if row_dict.get('epochs', '').isdigit() and row_dict.get('batch_size', '').isdigit():
                        row_dict['learning_rate'] = '0.001'
                    else:
                        row_dict['learning_rate'] = 'N/A' # それ以外はN/A
                
                # 修正された行をリストに変換して追加
                new_row = [row_dict.get(col, '') for col in header]
                migrated_data.append(new_row)

    except (IOError, csv.Error) as e:
        print(f"エラー: ログファイルの読み込みまたは解析に失敗しました: {e}", file=sys.stderr)
        return

    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(migrated_data)
        print(f"--- ログデータの移行が完了しました: {os.path.basename(output_file_path)} ---")
    except IOError as e:
        print(f"エラー: 移行済みログファイルの書き出しに失敗しました: {e}", file=sys.stderr)

if __name__ == '__main__':
    input_log_path = os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log.csv")
    output_log_path = os.path.join(PROJECT_ROOT, "logs", "pipeline_experiment_log_migrated.csv")
    migrate_log_data(input_log_path, output_log_path)
