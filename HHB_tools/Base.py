
from detector.ultralytics_main.ultralytics.data import loaders
    
def HHB_dataload(kargs):
    return iter(loaders.LoadImages(path=kargs['data_PATH']))

def HHB_visualize(track,kargs):
    pass

def HHB_save(track,kargs):
    pass

def HHB_imgpreprocess(frame,kargs):
    if isinstance(frame,(tuple,list)):
        frame = frame[1][0]
    return frame

def HHB_boxpostprocess(box,kargs):
    if isinstance(box,(tuple,list)):
        box=box[0].boxes.data
    box = box[box[:,-1]==0]
    return box

@staticmethod
def HHB_trackpostprocess(track,kargs):
    return track

