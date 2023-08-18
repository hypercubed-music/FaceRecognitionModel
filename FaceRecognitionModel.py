import tensorflow as tf
from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
import cv2
import os
from itertools import product, permutations
import random
from tqdm import tqdm
import numpy as np
from modules.utils import load_yaml, l2_norm
from mtcnn import MTCNN
from matplotlib import pyplot as plt
from tqdm import tqdm

class FaceRecognitionModel:
    def __init__(self, model_weights, config_path):
        '''
        Initialized the model

        model_weights: Path to model weights
        config_path: Path to config file
        '''

        tf.keras.utils.disable_interactive_logging()
        cfg = load_yaml(config_path)

        base_model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True,
                         logist_scale=64)
        ckpt_path = tf.train.latest_checkpoint(model_weights)
        if ckpt_path is not None:
            print("[*] load ckpt from {}".format(ckpt_path))
            base_model.load_weights(ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(ckpt_path))
            exit()

        self.model = tf.keras.models.Sequential(base_model.layers[:-2])
        self.detector = MTCNN()

    def get_video_feature_embedding(self, video, rate=5):
        '''
        Gets the average feature embedding for a video

        video: Video data, as a byte array
        rate: Rate to sample frames, in seconds (default 0.25 seconds)
        '''

        # yes, I know this is stupid, but cv2 won't work with byte array variables
        with open("temp.mp4", "w") as video_file:
            video_file.write(video)

        video = cv2.VideoCapture("temp.mp4")
        video_frames = []
        video_embs = []
        
        # extract frames
        print("Extracting frames...")
        success,image = video.read()
        count = 0
        while success:
            video.set(cv2.CAP_PROP_POS_FRAMES,(count*rate))    # added this line 
            success,image = video.read()
            video_frames.append(image)
            count += 1

        # get embeddings for frames
        print("Generating embedding...")
        for frame in tqdm(video_frames[:-1]):
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                emb = self.get_image_feature_embedding(frame)
                video_embs.append(emb)
            except Exception as e:
                print(str(e))
                continue
        
        os.remove("temp.mp4")

        return np.mean(video_embs, axis=0)


    def get_video_feature_embedding_filepath(self, video_path, rate=5, cache_embs=True):
        '''
        Gets the average feature embedding for a video

        video_path: Path to video file
        rate: Rate to sample frames, in seconds (default 0.25 seconds)
        '''

        video = cv2.VideoCapture(video_path)
        video_frames = []
        video_embs = []
        
        # extract frames
        print("Extracting frames...")
        success,image = video.read()
        count = 0
        while success:
            video.set(cv2.CAP_PROP_POS_FRAMES,(count*rate))    # added this line 
            success,image = video.read()
            video_frames.append(image)
            count += 1

        print(f"Extracted {count} frames")
        # get embeddings for frames
        print("Generating embedding...")
        for frame in tqdm(video_frames[:-1]):
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                emb = self.get_image_feature_embedding(frame)
                video_embs.append(emb)
            except Exception as e:
                print(str(e))
                continue

        if cache_embs:
            self.cache[video_path] = np.mean(video_embs, axis=0)
        
        return np.mean(video_embs, axis=0)

    def videos_have_same_person(self, video_1, video_2, threshold=1.5):
        '''
        Compares two videos and returns if the model sees that they have the same people
        '''

        emb_1 = self.get_video_feature_embedding(video_1)
        emb_2 = self.get_video_feature_embedding(video_2)
        return self.compare_embeddings(emb_1, emb_2, threshold)

    def video_files_have_same_person(self, video_1, video_2, threshold=1.5):
        '''
        Compares two videos and returns if the model sees that they have the same people
        '''

        emb_1 = self.get_video_feature_embedding_filepath(video_1)
        emb_2 = self.get_video_feature_embedding_filepath(video_2)
        return self.compare_embeddings(emb_1, emb_2, threshold)
    
    def get_image_feature_embedding(self, img):
        '''
        Gets the embedding of an image
        '''
        tf.keras.utils.disable_interactive_logging()

        detected_face = self.detector.detect_faces(img)[0]
        bbox = detected_face['box']
        cropped_face = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        resized_img = cv2.resize(cropped_face, (112, 112), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.
        resized_img = np.expand_dims(resized_img, 0)
        return l2_norm(self.model(resized_img))

    def imgs_have_same_person(self, img_1, img_2, threshold=1.5):
        '''
        Compares two images and returns if the model sees that they have the same people
        '''

        emb_1 = self.get_image_feature_embedding(img_1)
        emb_2 = self.get_image_feature_embedding(img_2)
        return self.compare_embeddings(emb_1, emb_2, threshold)

    def compare_embeddings(self, emb_1, emb_2, threshold=1.5):
        diff = np.subtract(emb_1, emb_2)
        dist = np.sum(np.square(diff), 1)

        return dist < threshold

if __name__ == "__main__":
    model = FaceRecognitionModel("./checkpoints/arc_patch_convnext_actual","./configs/arc_patch_convnext.yaml")

    print(model.videos_have_same_person("my_video.mp4", "my_other_video.mp4"))