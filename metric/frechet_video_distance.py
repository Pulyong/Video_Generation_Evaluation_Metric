import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video
from glob import glob
from tqdm import tqdm
import json
import scipy

'''
Based on https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py
'''

class FVD_Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        
        ret = {}
        
        video,_,vinfo = read_video(self.file_path[idx])
        gt = self.file_path[idx].split('/')[-1].split('-')[0]
    
        # get center 16 frame
        start = video.size()[0] // 2 - 8
        end = start + 16
        
        video_16 = video[start:end]
        video_16 = video_16.permute((3,0,1,2))     
        
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
    # model download
    model_path = os.path.join(save_path,"i3d_kinetics-400.pt")

    if model_name == 'I3D':
        if not os.path.isfile(model_path):
            import requests
            '''
            https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py
            '''

            url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'

            response = requests.get(url)
            with open(model_path, "wb") as file:
                file.write(response.content)

            print("Download success")
        model = torch.jit.load(model_path)
        model_kwargs = dict(rescale=True, resize=True, return_features=True)
    else:
        NotImplementedError('Not Implemented model!')

    return model, model_kwargs

def inference(model, dataloader, feature_stats, device, model_kwargs):
    
    stats = feature_stats

    model.to(device)
    model.eval()
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            video = batch['video'].to(device)
            B,T,C,H,W = video.size()
            
            features = model(video, **model_kwargs)

            stats.append_torch(features)
            
    return stats

def compute_fvd(gen_stats,real_stats):
    
    # compute fid
    mu_gen, sigma_gen = gen_stats.get_mean_cov()
    mu_real, sigma_real = real_stats.get_mean_cov()

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fvd = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    
    return fvd

def FVD(gen_video_path = None, real_video_path = None ,model_save_path = None, frame=16, device='cuda'):
    model_name = 'I3D'
    data_style = 'Kentics-400'

    
    model, model_kwargs = load_model(model_name, save_path = model_save_path) # B,C,T,H,W
    
    gen_stats = FeatureStats()
    gen_dataset = FVD_Dataset(glob(gen_video_path+'/*')) 
    gen_dataloader = DataLoader(gen_dataset,batch_size=64,num_workers=4,shuffle=False)

    real_stats = FeatureStats()
    real_dataset = FVD_Dataset(real_video_path)
    real_dataloader = DataLoader(real_dataset,batch_size=64,num_workers=4,shuffle=False) 

    gen_stats = inference(model=model, dataloader=gen_dataloader, feature_stats = gen_stats, device = device,model_kwargs=model_kwargs)
    real_stats = inference(model=model, dataloader=real_dataloader, feature_stats = real_stats, device = device, model_kwargs=model_kwargs)

    score = compute_fvd(gen_stats, real_stats)
    print(f"Frechet Video Distance estimated by {model_name}: {score}")
    

if __name__ =='__main__':
    
    gen_video_path = '/data/vilau_bench/UCF-101/generated/16frame/bench1'

    with open('/data/vilau_bench/UCF-101/video-lavit_sampling/20_sampled_vid_prompt_pair.json','r',encoding='utf-8') as file:
        data = json.load(file)

    path_list = []
    prompt_list = []
    for path in data:
        path_list += data[path]['video_path']
        prompt_list += data[path]['prompt']

    FVD(gen_video_path, path_list,'/data/video_bench_ckpt/FVD/I3D',device='cuda:5')