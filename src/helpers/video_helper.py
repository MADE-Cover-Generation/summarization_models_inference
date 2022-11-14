from os import PathLike
from pathlib import Path
import logging

import cv2
import numpy as np
from PIL import Image
from numpy import linalg
from torchvision import transforms
import time

import onnx
import onnxruntime

ort_session = onnxruntime.InferenceSession("../models/feature_extraction_googlenet.onnx")

from kts.cpd_auto import cpd_auto

SAMPLE_RATE = 30
MAX_SHOT_LENGTH = 15 * 30 # 15 frames and 30 sec
MIN_SHOT_LENGTH = 5 * 30  # 5 frames
N_FRAMES_IN_ONE_PART = 300 # for KTS

logger = logging.getLogger()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class FeatureExtractor(object):
    def __init__(self):
        # onnx
        start_time = time.time()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = onnx.load("../models/feature_extraction_googlenet.onnx")
        onnx.checker.check_model(self.model)

        logger.info(f"FeatureExtractor __init__ time: {time.time() - start_time}")

    def run(self, img: np.ndarray) -> np.ndarray:
        # onnx
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch)}
        ort_outs = ort_session.run(None, ort_inputs)

        feat = ort_outs[0].reshape(1024,)

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat /= linalg.norm(feat) + 1e-10
        return feat


class VideoPreprocessor(object):
    def __init__(self, sample_rate: int) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        fps = cap.get(cv2.CAP_PROP_FPS)

        features = []
        n_frames = 0

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"length: {length}")

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                feat = self.model.run(frame)
                features.append(feat)

            n_frames += 1

            if n_frames % 10000 == 0:
                logger.info(f"Processed {n_frames} / {length} n_frames")

        logger.info(f'Features are extracted: {time.time() - start_time}')
        cap.release()

        features = np.array(features)
        return n_frames, features, fps

    def kts(self, n_frames, features, divide_features=True, divide_intervals_shorter=True):
        start_time = time.time()
        seq_len = len(features)
        logger.info(f"seq_len: {seq_len}")
        picks = np.arange(0, seq_len) * self.sample_rate

        # compute change points using KTS
        if (divide_features) and (seq_len > N_FRAMES_IN_ONE_PART):
            cps_list = []

            n_parts = int(seq_len / N_FRAMES_IN_ONE_PART)
            logger.info(f"n parts: {n_parts}")
            logger.info(f"N_FRAMES_IN_ONE_PART: {N_FRAMES_IN_ONE_PART}")

            step = int(seq_len / n_parts)

            threshold = np.arange(0, seq_len, step, dtype=int)

            boundary_list = []
            for boundary in threshold:
                boundary_list.append((boundary, boundary + step))

            X_list = [features[boundary: boundary + step] for boundary in threshold]

            for X_part, n in zip(X_list, range(n_parts)):
                K = np.dot(X_part, X_part.T)
                cps, _ = cpd_auto(K, len(X_part) - 1, 1)
                cps = cps + n * step
                cps_list.append(cps)

            change_points = np.concatenate(cps_list)

        else:
            kernel = np.matmul(features, features.T)
            change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)

        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1] + self.sample_rate
        end_frames = change_points[1:] - self.sample_rate
        change_points = np.vstack((begin_frames, end_frames)).T

        change_points = np.array([cp for cp in change_points if cp[1] - cp[0] > self.sample_rate])

        divide_intervals_start = time.time()
        if divide_intervals_shorter:
            change_points = self.divide_intervals(change_points)
        logger.info(f"divide_intervals time: {time.time() - divide_intervals_start}")

        n_frame_per_seg = end_frames - begin_frames
        logger.info(f"VideoPreprocessor kts time: {time.time() - start_time}")
        return change_points, n_frame_per_seg, picks

    def process_interval(self, interval, max_shot_length=MAX_SHOT_LENGTH,
                         min_shot_length=MIN_SHOT_LENGTH):

        start_interval = interval[0]
        end_interval = interval[1]

        if end_interval - start_interval <= max_shot_length:
            return [interval]

        divided_interval = []
        n_short_intervals = int((end_interval - start_interval - 1) / max_shot_length)
        start = start_interval

        for i in range(1, n_short_intervals):
            divided_interval.append([start, start + max_shot_length])
            start = start + max_shot_length + 1

        if (end_interval - start) <= max_shot_length:
            # one big interval
            if start <= end_interval:
                divided_interval.append([start, end_interval])

        elif (end_interval - start) <= max_shot_length + min_shot_length:
            # one big interval and 1 short interval
            new_step = int((max_shot_length + min_shot_length) / 2)
            divided_interval.append([start, start + new_step])
            divided_interval.append([start + new_step + 1, end_interval])

        else:
            # two big intervals
            divided_interval.append([start, start + max_shot_length])
            divided_interval.append([start + max_shot_length + 1, end_interval])

        return divided_interval

    def divide_intervals(self, change_points, max_shot_length=MAX_SHOT_LENGTH):
        final_intervals = []

        for interval in change_points:
            processed = self.process_interval(interval, max_shot_length=max_shot_length)
            final_intervals.append(processed)

        final_intervals = np.concatenate(final_intervals)
        return final_intervals

    # assert np.allclose(divide_intervals([[10, 50]]),
    #                    np.array([[10, 25], [26, 41], [42, 50]]))
    # assert np.allclose(divide_intervals([[10, 50], [51, 55]]),
    #                    np.array([[10, 25], [26, 41], [42, 50], [51, 55]]))
    # assert np.allclose(divide_intervals([[10, 50], [51, 66]]),
    #                    np.array([[10, 25], [26, 41], [42, 50], [51, 66]]))
    # assert np.allclose(divide_intervals([[10, 50], [51, 66], [67, 97]]),
    #                    np.array([[10, 25], [26, 41], [42, 50], [51, 66], [67, 82], [83, 97]]))

    def run(self, video_path: PathLike):
        n_frames, features, fps = self.get_features(video_path)
        cps, nfps, picks = self.kts(n_frames, features)
        return n_frames, features, cps, nfps, picks, fps