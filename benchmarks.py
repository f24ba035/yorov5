import csv
import os
import time
from datetime import datetime

import cv2
import schedule
import torch

# YOLOv5リポジトリのルートディレクトリをパスに追加（相対パスの場合）
# detect.py と同じ階層にこのスクリプトを置く場合、多くは不要ですが、念のため。
# 必要に応じてsys.path.append(os.path.abspath(''))などを追加

# YOLOv5モデルのロード
# 'yolov5s' は事前学習済みモデルの名前。GPUが利用可能なら自動的に使われます。
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.eval()  # 推論モードに設定

# COCOデータセットにおける「車 (car)」のクラスIDは 2
# 別のクラスを検出したい場合は、names = model.names でクラス名リストを取得し、対応するIDを確認してください。
CAR_CLASS_ID = 2

# 結果を保存するCSVファイルパス
# YOLOv5リポジトリの 'runs/parking_counts' のような場所を指定すると良いでしょう
OUTPUT_CSV_PATH = "runs/parking_counts/car_counts.csv"


def setup_csv_file():
    """CSVファイルが存在しない場合にヘッダーを作成する."""
    if not os.path.exists(os.path.dirname(OUTPUT_CSV_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH))

    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Car Count"])


def detect_and_count_cars():
    """画像を読み込み、車を検出し、その数をカウントしてCSVに記録する関数."""
    print(f"{datetime.now()}: 車の台数検知を開始します...")

    # ここに駐車場の画像を読み込む処理を書く
    # 例1: 固定の画像ファイル
    image_path = "data/images/parking_lot_example.jpg"  # ここを実際の駐車場の画像パスに置き換える
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像ファイルが見つかりませんまたは読み込めません: {image_path}")
        return

    # 例2: Webカメラからの取得 (リアルタイム監視の場合)
    # cap = cv2.VideoCapture(0) # 0は通常の内蔵Webカメラ
    # ret, img = cap.read()
    # cap.release()
    # if not ret:
    #     print("エラー: Webカメラから画像をキャプチャできませんでした。")
    #     return

    # 推論の実行
    # results.xyxy[0] は、検出された物体ごとの [x1, y1, x2, y2, confidence, class_id] のテンソル
    results = model(img)

    # 車の数をカウント
    car_count = 0
    detections = results.xyxy[0]  # 最初の画像（バッチサイズ1の場合）の検出結果
    if detections is not None and len(detections) > 0:
        # 信頼度閾値でフィルタリング (detect.pyのデフォルトは0.25)
        # ここでは0.50に設定してみる
        filtered_detections = detections[detections[:, 4] > 0.50]

        # クラスIDが車のものだけをカウント
        for *xyxy, conf, cls_id in filtered_detections:
            if int(cls_id) == CAR_CLASS_ID:
                car_count += 1

    # 結果をCSVに記録
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, car_count])

    print(f"{timestamp}: 検出された車の台数: {car_count}台。結果をCSVに記録しました。")


# --- メイン処理 ---
if __name__ == "__main__":
    setup_csv_file()  # CSVファイルの準備

    # 10分毎に detect_and_count_cars 関数を実行するようにスケジュール
    print("駐車場監視を開始しました。10分ごとに車の台数を検知します。")
    schedule.every(10).minutes.do(detect_and_count_cars)

    while True:
        schedule.run_pending()  # スケジュールされたタスクを実行
        time.sleep(1)  # 1秒待機してCPU負荷を軽減
