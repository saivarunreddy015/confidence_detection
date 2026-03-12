import os
import shutil

# CHANGE THESE PATHS
source = "C:/Users/VARUN/Downloads/archive/train"
destination = "C:/Users/VARUN/Downloads/confidence_dataset/train"

mapping = {
    "happy": "confident",
    "neutral": "confident",
    "surprise": "confident",
    "angry": "not_confident",
    "sad": "not_confident",
    "fear": "not_confident",
    "disgust": "not_confident"
}

for emotion in os.listdir(source):

    emotion_path = os.path.join(source, emotion)

    if emotion in mapping:

        new_label = mapping[emotion]
        target_path = os.path.join(destination, new_label)

        os.makedirs(target_path, exist_ok=True)

        for img in os.listdir(emotion_path):

            src = os.path.join(emotion_path, img)
            dst = os.path.join(target_path, img)

            shutil.copy(src, dst)

print("Dataset conversion completed!")