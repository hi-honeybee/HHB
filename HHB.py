import importlib
import yaml

with open('refer_path.yaml') as f:
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

if __name__=='__main__':
    # input: data_PATH
    # output: iter(tensor[h,w,c])
    video_iter = DataLoader.HHB_dataload(kargs)

    # input: weight_PATH
    # output: object - for detection
    detector = DetectorLoader.HHB_detectorload(kargs)

    # input: None
    # output: object - for tracking
    tracker = TrackerLoader.HHB_trackerload(kargs)

    for kargs['i'],frame in enumerate(video_iter):
        # input: tensor[h,w,c]
        # output: tensor[h,w,c]
        frame = Iprp.HHB_imgpreprocess(frame,kargs)
        
        # input: tensor[h,w,c]
        # output: tensor[:, 6] where 6 is (x,y,w,h,scr,cls)
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
        Visualizer.HHB_visualize(track,kargs)

        # input: list(STrack)
        Save.HHB_save(track,kargs)

        print(kargs['i'],)