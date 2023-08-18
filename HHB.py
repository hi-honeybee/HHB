import importlib
import yaml
import torch

with open('refer_path5.yaml') as f:
    refer_path = yaml.load(f, Loader=yaml.FullLoader)

DataLoader = importlib.import_module(refer_path['DataLoader'])
Iprp = importlib.import_module(refer_path['ImagPreprocessor'])
DetectorLoader = importlib.import_module(refer_path['DetectorLoader'])
Dpop = importlib.import_module(refer_path['DetectionPostprocessor'])
TrackerLoader = importlib.import_module(refer_path['TrackerLoader'])
Tpop = importlib.import_module(refer_path['TrackerPostprocessor'])
Visualizer = importlib.import_module(refer_path['Visualizer'])
Save = importlib.import_module(refer_path['Save'])
# data_PATH, weight_PATH + additional arguments
kargs=refer_path['kargs']


import os
import time
def cycle(date='0807'):
    while True:
        for i in range(150):
            print(i, 'start')
            kargs['data_PATH'] = 'datasets/'+date+'/'+str(i)+'.mp4'
            if not os.path.exists(kargs['data_PATH']): 
                print(i,'not exists')
                continue
            kargs['track-PATH']='track_result/'+date+'/'+str(i)
            if not os.path.exists('track_result/'+date): 
                os.mkdir('track_result/'+date)
            if os.path.exists(kargs['track-PATH']+'.pickle'):
                print(i,'pass')
                continue
            HHB()
            print(i,'finish')
        time.sleep(600)

def HHB():
    # input: data_PATH
    # output: iter(tensor[h,w,c])
    print(kargs['data_PATH'] )
    video_iter = DataLoader.HHB_dataload(kargs)
    
    # input: weight_PATH
    # output: object - for detection
    detector = DetectorLoader.HHB_detectorload(kargs)

    # input: None
    # output: object - for tracking
    tracker = TrackerLoader.HHB_trackerload(kargs)

    tracks=[]
    for kargs['i'],frame in enumerate(video_iter):
        # input: tensor[h,w,c]
        # output: tensor[h,w,c]
        frame = Iprp.HHB_imgpreprocess(frame,kargs)
        
        # input: tensor[h,w,c]
        # output: tensor[:, 6] where 6 is (t,l,b,r,scr,cls)
        box = detector.HHB_detect(frame,kargs)
        
        # input: tensor[:, 6]
        # output: tensor[:, 6]
        box = Dpop.HHB_boxpostprocess(box,kargs)
        
        # input: tensor[:, 6]
        # output: list(STrack)
        track = tracker.HHB_track(box,kargs)

        # input: list(STrack)
        # output: list(STrack)
        track = Tpop.HHB_trackpostprocess(track,kargs)

        # input: list(STrack)
        # Visualizer.HHB_visualize(track,kargs)
        
        tracks.append(track)

    # input: list(STrack)
    Save.HHB_save(tracks,kargs)

        
if __name__=='__main__':
    cycle()