# HHB/tracker/SparseTrack_main/ 에 위치


import pickle
from tracker.sparse_tracker import *

## Load pickle
with open("../../track_result/track1.pickle","rb") as fr:
    data = pickle.load(fr)
print(data)
print('tlwh:',data[0].tlwh)
print('track_id:',data[0].track_id)
print('score:',data[0].score)