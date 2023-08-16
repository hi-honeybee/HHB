from HHB_tools import frameslicer
from HHB_tools import boxhandler3

def HHB_boxpostprocess(box,kargs):
    box=frameslicer.HHB_imgrecombine(box,kargs)
    return boxhandler3.HHB_boxpostprocess(box,kargs)
