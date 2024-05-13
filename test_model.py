from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

img_width = 256
img_height = 256

classes = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"] # Sorted alphabetically

model = load_model("chess_best_model.keras")

def prepare_img(img_file):
    img = load_img(img_file, target_size=(img_height, img_width))
    img_res = img_to_array(img)
    img_res = np.expand_dims(img_res, axis=0)
    img_res = img_res/255.0
    return img_res


def predict_class(img_class):
    TEST_DIR = Path(f'test/{img_class}')
    prediction = {"Bishop": 0, "King": 0, "Knight": 0, "Pawn": 0, "Queen": 0, "Rook": 0}
    for img_path in TEST_DIR.iterdir():
        if img_path.is_file():
            img_for_model = prepare_img(img_path)
            res_arr = model.predict(img_for_model, batch_size=32, verbose=1)
            answer = np.argmax(res_arr, axis=1)
            text = classes[answer[0]]
            prediction[text] += 1

    print(f'Actual class: {img_class}')
    print("Predicted class:")
    for key, value in prediction.items():
        print(key, value)

for img_class in classes:
    predict_class(img_class)

