import torch
import numpy as np
import cv2
from torchvision import transforms
from modules.layers_maniqa.maniqa import MANIQA
import piq
import logging

#workaround for https://github.com/facebookresearch/fairseq/issues/2510
from torch import Tensor
import torch.nn.functional as F

class GELU(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

torch.nn.modules.activation.GELU = GELU

logger = logging.getLogger()

class ManiqaHelper():

    CROP_SIZE = 224

    def __init__(self, model_path: str, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.model = torch.nn.DataParallel(MANIQA())
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.best_frames_indexes = [] 

    def __prepare(self, img):
        d_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        sample = {
            'd_img_org': d_img,
            'd_name': ""
        }
        transform = transforms.Compose([Normalize(0.5, 0.5), ToTensor()])
        sample = transform(sample)
        return sample

    def __random_crop(self, d_img):
        c, h, w = d_img.shape
        top = np.random.randint(0, h - self.CROP_SIZE)
        left = np.random.randint(0, w - self.CROP_SIZE)
        d_img_org = self.__crop_image(top, left, self.CROP_SIZE, img=d_img)
        d_img_org = d_img_org[None, :]
        return d_img_org

    def __crop_image(self, top, left, patch_size, img=None):
        tmp_img = img[:, top:top + patch_size, left:left + patch_size]
        return tmp_img
        
    def __brisque_score(self, frame):
        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        shot = (
            torch.tensor(frame_).permute(2, 0, 1) / 255.0
        )
        brisque_score = piq.brisque(
                shot[None, ...].to(self.device), data_range=1.0, reduction="none"
            ).item()
        return brisque_score 

    def inference(self, index: int, frame):
        brisque_score = self.__brisque_score(frame)
        prepared = self.__prepare(frame)
        img = prepared['d_img_org'].to(self.device)
        img = self.__random_crop(img)
        res = self.model(img).item()
        
        return (index, brisque_score, res)

    def find_best_frame_index(self, indexed_scores: list) -> int:
        median = np.median([x[1] for x in indexed_scores])
        filtered_scores = [tup for tup in indexed_scores if tup[1] < median]
        index, brisque_score, maniqa_score = max(filtered_scores, key=lambda tup: tup[2])
        logger.info(f"best_frame_index: {index}, brisque_score: {brisque_score}, maniqa_score: {maniqa_score}")
        self.best_frames_indexes.append(index)

    def get_best_frames_indexes(self) -> list:
        return self.best_frames_indexes

    def clear(self):
        self.best_frames_indexes = []


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        d_name = sample['d_name']

        d_img = (d_img - self.mean) / self.var

        sample = {'d_img_org': d_img, 'd_name': d_name}
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        d_name = sample['d_name']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'd_name': d_name
        }
        return sample