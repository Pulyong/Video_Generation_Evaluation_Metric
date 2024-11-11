import torch
import pickle
import numpy as np
import gdown
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video
from glob import glob
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

class CLIPSIM_Dataset(Dataset):
    def __init__(self, file_path, processor, frame=16):
        self.file_path = file_path
        self.processor = processor
        self.frame = frame

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        
        ret = {}
        video,_,vinfo = read_video(self.file_path[idx])
        gt = self.processor(text = self.file_path[idx].split('/')[-1].split('-')[0],return_tensors="pt",padding='max_length',max_length=77)['input_ids'].squeeze()
        
        start = (video.size()[0] // 2) - (self.frame // 2) # C,H,W,T
        end = start + self.frame
        
        video_16 = video[start:end]
        #transformed_frames = [self.transform(images=frame, return_tensors="pt")['pixel_values'] for frame in video_16]
        video_16 = self.processor(images=video_16, return_tensors="pt")['pixel_values'] # T,C,H,W
        
        ret['video'] = video_16
        ret['gt'] = gt

        return ret

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
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16",cache_dir = save_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16",cache_dir = save_path)
    
    return model, processor

def inference(model, dataloader, device):

    model.to(device)
    model.eval()
    
    clip_score = 0
    num_items = 0
    for batch in tqdm(dataloader):
        with torch.no_grad():
            video = batch['video'].to(device)
            text = batch['gt'].to(device)
            B,T,C,H,W = video.size()

            video_features = model.get_image_features(video.reshape(-1,C,H,W))
            text_features = model.get_text_features(text)

            # normalize features
            video_features = video_features / video_features.norm(dim=1, keepdim=True).to(torch.float32) # B*T,512
            text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)

            text_features = text_features.repeat_interleave(repeats=T,dim=0) # B*T,512

            score = (text_features * video_features).sum(dim=1)
            score = score.reshape(B,T)
            score = score.mean(dim=1).sum()
            clip_score += score
            num_items += B

    return clip_score / num_items

def CLIPSIM(gen_video_path = None, model_save_path = None, frame=16, device='cuda'):
    model_name = 'ViT-B16'
    data_style = ''

    model, processor = load_model(model_name, model_save_path)

    dataset = CLIPSIM_Dataset(glob(gen_video_path + '/*'),processor,frame=frame)
    dataloader = DataLoader(dataset,batch_size=16,num_workers=4,shuffle=False)

    score = inference(model, dataloader, device)
    print(f"Clip Similarity estimated by {model_name}: {score}")

if __name__ == '__main__':
    CLIPSIM(None,'/data/video_bench_ckpt/CLIPSIM',frame=8,device='cuda:5')