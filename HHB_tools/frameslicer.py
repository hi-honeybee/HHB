import torch

def HHB_imgpreprocess(frame,kargs):
    frames=[]
    if isinstance(frame,(tuple,list)):
        frame = frame[1][0]
        
    kargs['orig_img']=frame 
    frames.append(frame[:640,:720])
    frames.append(frame[440:1080,:720])
    frames.append(frame[:640,600:1320])
    frames.append(frame[440:1080,600:1320])
    frames.append(frame[:640,1200:1920])
    frames.append(frame[440:1080,1200:1920])
    return frames

def HHB_imgrecombine(box,kargs):
    boxes=[]
    if isinstance(box,(tuple,list)):
        boxes.append(box[0].boxes.data)
        boxes.append(box[1].boxes.data+torch.tensor([0,440,0,440,0,0],device='cuda:0'))
        boxes.append(box[2].boxes.data+torch.tensor([600,0,600,0,0,0],device='cuda:0'))
        boxes.append(box[3].boxes.data+torch.tensor([600,440,600,440,0,0],device='cuda:0'))
        boxes.append(box[4].boxes.data+torch.tensor([1200,0,1200,0,0,0],device='cuda:0'))
        boxes.append(box[5].boxes.data+torch.tensor([1200,440,1200,440,0,0],device='cuda:0'))
    box= torch.concat(boxes,axis=0)
    return box
    

    