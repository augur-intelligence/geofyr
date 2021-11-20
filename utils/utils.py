import torch
from torch import cos, sin, arccos,arcsin, mean
import os
import gcsfs

# Storage
CRED_PATH = "/".join([os.getcwd(), 'utils', 'storage.json'])
fs = gcsfs.GCSFileSystem(token=CRED_PATH)
storage_options={"token": CRED_PATH}
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CRED_PATH

# Constants
PI = torch.acos(torch.zeros(1)).item() * 2
EARTH_RADIUS = 6371
DEG2RAD = PI/180

class GEODataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)
    
def haversine_dist(logits, labels):
    ## phi = lat = index 0, lambda = lon = index 1
    labels = DEG2RAD * labels
    d_sigma = 2 * (
        arcsin(
            sqrt(
                square(
                    sin(
                        (logits[:,0]-labels[:,0]) / 2 )
                ) 
                + 
                cos(logits[:,0]) 
                * 
                cos(labels[:,0]) 
                * 
                square(
                    sin(
                        (logits[:,1]-labels[:,1]) / 2 )
                ) 
            )
        )
    )
    hav_dist = EARTH_RADIUS * d_sigma
    hav_dist_mean = mean(hav_dist)
    return hav_dist_mean