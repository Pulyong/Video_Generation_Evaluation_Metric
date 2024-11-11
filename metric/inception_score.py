import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video
from tqdm import tqdm

'''
Based on https://github.com/universome/stylegan-v/blob/master/src/metrics/video_inception_score.py
'''
class FeatureStats:
    '''
    Class For Saving Features
    https://github.com/universome/stylegan-v/blob/master/src/metrics/metric_utils.py#L63
    '''
    def __init__(self):
        self.num_items = 0
        self.num_features = None
        self.all_features = None
    
    def set_num_features(self, num_features):
        '''
        init raw_mean,raw_cov array
        '''
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
        
    def append(self, x):
        '''
        append features to raw_mean,raw_cov
        '''
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        
        self.set_num_features(x.shape[1])
        self.all_features.append(x)
        self.num_items += x.shape[0]
        
        
    def append_torch(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        self.append(x.detach().cpu().numpy())
        
    def get_all(self):
        return np.concatenate(self.all_features, axis=0)

class UCF101_Dataset(Dataset):
    def __init__(self, file_path, frame = 16):
        self.file_path = file_path
        self.transform = transforms.Compose([
            transforms.Resize((256, 256))
            ])
        self.frame = frame

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        
        ret = {}
        
        video,_,vinfo = read_video(self.file_path[idx])
        
        # get center n frame
        
        start = (video.size()[0] // 2) - (self.frame // 2) # C,H,W,T
        end = start + self.frame
        
        crop_video = video[start:end]
        crop_video = crop_video.permute((3,0,1,2)) # T,C,H,W
        crop_video = self.transform(crop_video)       
        
        ret['video'] = crop_video
        
        return ret

def load_model(model_name:str, save_path:str):
    
    if model_name == 'C3D':
        # model download
        model_path = os.path.join(save_path,"c3d_ucf101.pt")

        if not os.path.isfile(model_path):
            import requests
            
            url = "https://www.dropbox.com/s/jxpu7avzdc9n97q/c3d_ucf101.pt?dl=1"

            response = requests.get(url)
            with open(model_path, "wb") as file:
                file.write(response.content)

            print("Download success")

        else:   
            NotImplementedError('Not Implemented Data!')
            
        model = torch.jit.load(model_path)
    else:
        raise NotImplementedError('Not Implemented!')

    return model

def inference(model, dataloader, feature_stats, device):
    
    stats = feature_stats

    model.to(device)
    model.eval()
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            video = batch['video'].to(device)
            B,T,C,H,W = video.size()
            
            features = model(video)

            stats.append_torch(features)
            
    return stats

def compute_IS(stats):
    prob = stats.get_all()
    kl = prob * (np.log(prob) - np.log(np.mean(prob, axis=0, keepdims=True)))
    kl = np.mean(np.sum(kl, axis=1))
    kl = np.exp(kl)
    return kl

def IS(video_path = None, model_save_path = None, frame=16, device='cuda'):
    
    model_name = 'C3D'
    data_style = 'UCF101' # Only Support C3D trained on UCF101

    print(f"Model: {model_name}, Trained on: {data_style}")
    
    model = load_model(model_name, model_save_path)
    dataset = UCF101_Dataset(glob(video_path+'/*'), frame)
    dataloader = DataLoader(dataset,batch_size=64,num_workers=4,shuffle=False)
    
    stats = FeatureStats()
    
    stats = inference(model= model, dataloader=dataloader, feature_stats = stats, device = device)
    
    score = compute_IS(stats)
    print(f"Inception Score estimated by {model_name}: {score}")
    
if __name__=='__main__':
    IS(video_path='/data/vilau_bench/UCF-101/generated/16frame/bench1',model_save_path='/data/video_bench_ckpt/IS/C3D',device='cuda:5')