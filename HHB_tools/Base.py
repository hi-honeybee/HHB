
from detector.ultralytics_main.ultralytics.data import loaders
    
def HHB_dataload(kargs):
    return iter(loaders.LoadImages(path=kargs['data_PATH'],vid_stride=kargs.get('dataloader.vid_stride',1)))

def HHB_visualize(track,kargs):
    pass

def HHB_save(track,kargs):
    pass

def HHB_imgpreprocess(frame,kargs):
    if isinstance(frame,(tuple,list)):
        frame = frame[1][0]#[:640,:640]
    kargs['orig_img']=frame
    return frame

def HHB_boxpostprocess(box,kargs):
    if isinstance(box,(tuple,list)):
        box=box[0].boxes.data
    box = box[box[:,-1]==1]
    return box

@staticmethod
def HHB_trackpostprocess(track,kargs):
    return track

