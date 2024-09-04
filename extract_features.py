import shutil
import tqdm
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

path = "Data"

def video2frames(video , train_or_test="Training Videos"):
    # creating a temporary folder to store the frames extracted
    temp_path = os.path.join(path,"temp_images")
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)

    # video path for either testing or testing videos
    video_path = os.path.join(path, train_or_test, video)

    count = 0

    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cv2.imwrite(os.path.join(temp_path, 'frame%d.jpg' % count), frame)
        image_list.append(os.path.join(temp_path, 'frame%d.jpg' % count))
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return image_list

# Resizing each frame to (224,224)
def resize_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img

# Loading CNN Model to extract video features
def load_cnn_model():
    model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    out = model.layers[-2].output
    loaded_model = Model(inputs=model.input, outputs=out)
    return loaded_model


def extract_features(video, model):
    # Extracting the ID of the video
    video_id = video.split(".")[0]
    print(video_id)
    print(f'Processing video {video}')

    # Retrieving the list of frames
    image_list = video2frames(video, train_or_test="Training Videos")
    if len(image_list)<80:
      return

    # while len(image_list)<80:
    #   image_list = np.vstack([image_list, np.zeros_like(image_list[0])])

    # Uniform sampling and extracting 80 frames
    samples = np.round(np.linspace(0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]

    images = np.zeros((len(image_list), 224, 224, 3))
    for i in range(len(image_list)):
        img = resize_img(image_list[i])
        images[i] = img
    images = np.array(images)

    # Extracting video features and saving to numpy array
    fc_feats = model.predict(images, batch_size=128)
    img_feats = np.array(fc_feats)

    # cleanup
    shutil.rmtree(os.path.join(path,"temp_images"))
    return img_feats

def extract_feats_pretrained_cnn(train_or_test="Training Videos"):
    """
    saves the numpy features from all the videos
    """
    model = load_cnn_model()
    print('Model loaded')

    if not os.path.isdir(os.path.join(path, train_or_test, 'feat')):
        os.mkdir(os.path.join(path, train_or_test, 'feat'))

    video_list = os.listdir(os.path.join(path, train_or_test))

    
    for video in video_list:

        outfile = os.path.join(path, train_or_test, 'feat', video + '.npy')
        img_feats = extract_features(video, model)
        if img_feats is None:
          continue

        np.save(outfile, img_feats)


if __name__ == "__main__":
    extract_feats_pretrained_cnn()