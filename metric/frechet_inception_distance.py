import torch
import numpy as np
import clip
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video
from glob import glob
from tqdm import tqdm
import json
import scipy
from transformers import CLIPProcessor, CLIPModel

'''
Based on https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py
'''

class FID_Dataset(Dataset):
    def __init__(self, file_path, processor, frame = 16):
        self.file_path = file_path
        self.processor = processor
        self.frame = frame

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        
        ret = {}
        
        video,_,vinfo = read_video(self.file_path[idx])
        gt = self.file_path[idx].split('/')[-1].split('-')[0]
    
        # get center 16 frame
        start = (video.size()[0] // 2) - (self.frame // 2) # C,H,W,T
        end = start + self.frame
        
        video_16 = video[start:end]

        #transformed_frames = [self.transform(images=frame, return_tensors="pt")['pixel_values'] for frame in video_16]
        video_16 = self.processor(images=video_16, return_tensors="pt")['pixel_values'] # T,C,H,W

        #video_16 = video_16.permute((3,0,1,2)) ## 여기 고쳐
        ret['video'] = video_16
        ret['gt'] = gt
        
        return ret

class FeatureStats:
    '''
    Class For Saving Features
    https://github.com/universome/stylegan-v/blob/master/src/metrics/metric_utils.py#L63
    '''
    def __init__(self):
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None
    
    def set_num_features(self, num_features):
        '''
        init raw_mean,raw_cov array
        '''
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.num_items = 0
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)
        
    def append(self, x):
        '''
        append features to raw_mean,raw_cov
        '''
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2
        
        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        
        self.raw_mean += x.sum(axis=0)
        self.raw_cov += x.T @ x
        
        
    def append_torch(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        self.append(x.detach().cpu().numpy())
        
    def get_mean_cov(self):
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

def load_model(model_name:str, save_path:str):
    '''
    if model_name == 'ViT-B32':
        gdrive_id = '1j3pS3bdTXIYL56kpcpdMrrvPJVT90IY0'
        model_path = os.path.join(save_path,"ViT-B32","clip-ViT-B32.pkl")
    elif model_name == 'InceptionV3':
        gdrive_id = '1yDD9iqw3YYbkn2d7N8ciYu81widI-uEL'
        model_path = os.path.join(save_path,"InceptionV3","InceptionV3.pkl")
    else:
        raise NotImplementedError('not implemented!!')

    if not os.path.isfile(model_path):
        gdown.download(id=gdrive_id, output=model_path, quiet=False)        
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    '''
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",cache_dir = save_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = save_path)
    
    return model, processor

def inference(model, dataloader, feature_stats, device):
    
    stats = feature_stats

    model.to(device)
    model.eval()
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            video = batch['video'].to(device)
            B,T,C,H,W = video.size()
            
            features = model.get_image_features(video.reshape(-1,C,H,W))
            features = features.reshape(B,T,-1)

            stats.append_torch(features.mean(axis=1))
            
    return stats

def compute_fid(gen_stats,real_stats):
    
    # compute fid
    mu_gen, sigma_gen = gen_stats.get_mean_cov()
    mu_real, sigma_real = real_stats.get_mean_cov()

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    
    return fid

def FID(gen_video_path = None, real_video_path = None ,model_save_path = None, frame=16, device='cuda'):
    model_name = 'ViT-B32'
    data_style = ''

    model, processor = load_model(model_name, save_path = model_save_path) # B,C,T,H,W
    
    gen_stats = FeatureStats()
    gen_dataset = FID_Dataset(glob(gen_video_path+'/*'),processor,frame) 
    gen_dataloader = DataLoader(gen_dataset,batch_size=64,num_workers=4,shuffle=False)
    
    real_stats = FeatureStats()
    real_dataset = FID_Dataset(real_video_path,processor,frame)
    real_dataloader = DataLoader(real_dataset,batch_size=64,num_workers=4,shuffle=False) 

    gen_stats = inference(model=model, dataloader=gen_dataloader, feature_stats = gen_stats, device = device)
    real_stats = inference(model=model, dataloader=real_dataloader, feature_stats = real_stats, device = device)

    score = compute_fid(gen_stats, real_stats)
    print(f"Frechet Inception Distance estimated by {model_name}: {score}")
    

if __name__ =='__main__':
    
    gen_video_path = '/data/vilau_bench/MSR-VTT/generated/8frame/bench1'
    path_list = glob('/data/vilau_bench/MSR-VTT/MSR-VTT/TestVideo/*')
    FID(gen_video_path, path_list,'/data/video_bench_ckpt/FID',device='cuda:5',frame=8)