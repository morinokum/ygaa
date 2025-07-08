import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# このエージェントファイルの場所を基準にプロジェクトルートを特定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# デフォルト設定
DEFAULT_CONFIG = {
    "model_path": os.path.join(PROJECT_ROOT, "trained_models", "mnist_model_latest.keras"),
    "image_path": None
}

def preprocess_image(image_path):
    """
    単一の画像をモデルの入力形式に合わせて前処理する。
    """
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return None

    try:
        # 画像を開き、グレースケールに変換
        img = Image.open(image_path).convert('L')
        
        # MNISTの画像は「白地に黒文字」なので、色を反転させる（背景を黒、文字を白へ）
        img = ImageOps.invert(img)

        # 28x28ピクセルにリサイズ
        img = img.resize((28, 28))

        # NumPy配列に変換
        img_array = np.array(img)

        # 0-1の範囲に正規化
        img_array = img_array.astype("float32") / 255

        # モデルの入力形式 (batch_size, height, width, channels) に合わせる
        img_array = img_array[np.newaxis, ..., np.newaxis]
        
        return img_array
    except Exception as e:
        print(f"エラー: 画像の前処理中にエラーが発生しました: {e}")
        return None

def main(args, config):
    """
    学習済みモデルをロードし、指定された画像で推論を実行するエージェント。
    """
    print("Inference Agent: 開始")

    model_path = config.get("model_path")
    image_path = config.get("image_path")

    if not image_path:
        print("エラー: 推論する画像ファイルが指定されていません。--agent-set image_path=<path_to_image> で指定してください。")
        return

    # パスを絶対パスに変換
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.isabs(image_path):
        image_path = os.path.join(PROJECT_ROOT, image_path)

    # 1. モデルのロード
    if not os.path.exists(model_path):
        print(f"エラー: 学習済みモデルが見つかりません: {model_path}")
        return
    
    print(f"--- モデルをロード中: {model_path} ---")
    try:
        model = tf.keras.models.load_model(model_path)
        print("--- モデルのロードが完了しました ---")
    except Exception as e:
        print(f"エラー: モデルのロードに失敗しました: {e}")
        return

    # 2. 画像の前処理
    print(f"--- 画像を前処理中: {image_path} ---")
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return
    print("--- 画像の前処理が完了しました ---")

    # 3. 推論の実行
    print("--- 推論を実行中 ---")
    try:
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions)
        print("--- 推論が完了しました ---")

        # 4. 結果の表示
        print("\n========== 推論結果 ==========")
        print(f"入力画像: {os.path.basename(image_path)}")
        print(f"予測された数字: {predicted_class[0]}")
        print(f"確信度: {confidence:.2%}")
        print("==============================")

    except Exception as e:
        print(f"エラー: 推論の実行中にエラーが発生しました: {e}")

    print("Inference Agent: 終了")

if __name__ == '__main__':
    # このスクリプトが直接実行された場合のテスト用
    parser = argparse.ArgumentParser(description='MNIST推論エージェント')
    parser.add_argument('--model_path', type=str, default=DEFAULT_CONFIG["model_path"], help='学習済みモデルのパス')
    parser.add_argument('--image_path', type=str, required=True, help='推論する画像のパス')
    args = parser.parse_args()
    
    config = {"model_path": args.model_path, "image_path": args.image_path}
    main([], config)