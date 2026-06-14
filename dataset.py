import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset


class MaestroDataset(Dataset):
    def __init__(self, preprocessed_dir, split='train', sequence_length=128):
        self.files = glob.glob(os.path.join(preprocessed_dir, f"{split}_*.npz"))
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with np.load(file_path) as data:
            features = data['features']
            frames = data['frames']
            onsets = data['onsets']
            
        total_frames = features.shape[1]
        
        if total_frames > self.sequence_length:
            max_start = total_frames - self.sequence_length
            start_frame = np.random.randint(0, max_start)
            
            x_slice = features[:, start_frame:start_frame + self.sequence_length]
            y_frames = frames[:, start_frame:start_frame + self.sequence_length]
            y_onsets = onsets[:, start_frame:start_frame + self.sequence_length]
        else:
            pad_width = self.sequence_length - total_frames
            x_slice = np.pad(features, ((0, 0), (0, pad_width)), mode='minimum')
            y_frames = np.pad(frames, ((0, 0), (0, pad_width)), mode='constant')
            y_onsets = np.pad(onsets, ((0, 0), (0, pad_width)), mode='constant')

        # Convert to tensors
        x_tensor = torch.tensor(x_slice, dtype=torch.float32).unsqueeze(0) # [1, 88, SeqLen]
        y_frames_tensor = torch.tensor(y_frames, dtype=torch.float32)       # [88, SeqLen]
        y_onsets_tensor = torch.tensor(y_onsets, dtype=torch.float32)       # [88, SeqLen]
        
        return x_tensor, y_frames_tensor, y_onsets_tensor