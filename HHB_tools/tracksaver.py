import pickle
import torch

def HHB_save(track,kargs):
    ## Save pickle
    with open(kargs['track-PATH']+".pickle","wb") as fw:
        pickle.dump(track, fw)

# STrack to tensor
def HHB_trackpostprocess(track,kargs):
    tensor_track = torch.zeros(len(track),6)
    for i,t in enumerate(track):
        tensor_track[i,:4]=torch.tensor(t.tlwh)
        tensor_track[i,4]=torch.tensor(t.track_id)
        tensor_track[i,5]=torch.tensor(t.score)
    return tensor_track