import cv2
import numpy as np
import torch
import torchvision
import json


class BasketballActionDataset()
    def __init__(self, annotation_file, videos_dir='/dataset/examples', pose_data=False): 
        with open(annotation_file) as f:
            self.videos = list(json.load(f).items())

        self.videos_dir = videos_dir
        self.pose_data = pose_data

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
       v_id = self.video[idx][0]

        one_hot_result = np.squeeze(np.eye(10)[9 - self.videos[idx][1]])   
        if self.pose_data:
            item = {
                'v_id': v_id,
                'joints': np.load(f'{self.videos_dir}{id}.npy', allow_pickle=True),
                'action': torch.from_numpy(np.squeeze(np.eye(10)[9 - self.videos[idx][1]])),
                'class': self.videos[idx][1]
            }
        else:
            item = {
                'v_id': v_id,
                'video': torch.from_numpy(self.video_to_numpy(v_id)).float()
                'action': torch.from_numpy(np.squeeze(np.eye(10)[9 - self.videos[idx][1]])),
                'class': self.videos[idx][1]
            }
        return item

    def video_to_numpy(self, v_id):
        video = cv2.VideoCapture(f'{self.videos_dir}{id}.mp4')
        
        if not video.isOpened():
            raise Exception('Video Not Readable')

        v_frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret or not frame:
                break
            frame = np.asarray([frame[..., i] for i in range(frame.shape[-1])]).astype(float)
            v_frames.append(frame)

        video.release()
        return np.transpose(np.asarray(video_frames), (1,0,2,3)
