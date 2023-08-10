from ultralytics.engine.results import Results
import torch 

def HHB_visualize(track,kargs):
    predictor = kargs['Predictor']

    tracked_results=STrack2Results(track,kargs)

    predictor.write_results(tracked_results, kargs['detector.im'])
    predictor.save_preds(kargs['detector.vid_caps'])

    return

def STrack2Results(track,kargs):

    result = kargs['detector.result']
    orig_img = result[0].orig_img
    path = result[0].path
    names = result[0].names
    tracked_results = []
    preds = torch.zeros((len(track),7))
    for i,t in enumerate(track):
        preds[i,:4]=torch.tensor(tlwh2xyxy(t.tlwh))
        preds[i,4]=torch.tensor(t.track_id)
        preds[i,5]=torch.tensor(t.score)

    tracked_results.append(Results(orig_img=orig_img, path=path, names=names, boxes=preds))
    return tracked_results



def tlwh2xyxy(x):
    y=x.copy()
    y[2]=x[0]+x[2]
    y[3]=x[1]+x[3]
    return y