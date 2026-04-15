import cv2
import os

video_folders = {
    "real": "dataset_videos/real",
    "fake": "dataset_videos/fake"
}

output_folders = {
    "real": "dataset/real",
    "fake": "dataset/fake"
}

for label in video_folders:
    os.makedirs(output_folders[label], exist_ok=True)

    for video in os.listdir(video_folders[label]):

        video_path = os.path.join(video_folders[label], video)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % 10 == 0:
                frame_name = f"{video.split('.')[0]}_{frame_count}.jpg"
                save_path = os.path.join(output_folders[label], frame_name)
                cv2.imwrite(save_path, frame)

            frame_count += 1

        cap.release()

print("Frames extracted successfully")