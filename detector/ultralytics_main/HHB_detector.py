from ultralytics import YOLO

def HHB_detectorload(kargs):
    model = YOLO(kargs['weight-PATH'])
    model.ready4predict()
    kargs['Predictor']=model.predictor
    return model