import time

import cv2
import schedule
import torch

# YOLOモデルのロード
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.eval()


def detect_and_count_cars():
    # カメラから画像をキャプチャ (例: ファイルパス)
    img = cv2.imread("C:\\Users\\Owner\\Desktop\\machine_learning\\object_detection\\yolov5\\images.jpg")
    # cv2.VideoCapture
    # 上を使えばカメラからのリアルタイム映像も取得可能
    # img = cv2.VideoCapture(0).read()[1]  # カメラからの画像取得
    # YOLOで推論
    results = model(img)

    # 車のクラスID (COCOデータセットで car は 2)
    car_class_id = 2

    # 検出された車の数をカウント
    car_count = 0
    if results.xyxy[0] is not None:
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == car_class_id:
                car_count += 1

    # 現在時刻と車の台数を表示・記録
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time}: 検出された車の台数: {car_count}")
    # ここで結果をファイルやデータベースに保存する処理を追加


# 10分毎に detect_and_count_cars 関数を実行するようにスケジュール
schedule.every(10).minutes.do(detect_and_count_cars)

while True:
    schedule.run_pending()
    time.sleep(1)
