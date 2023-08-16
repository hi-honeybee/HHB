import sys
sys.path.append(sys.path[0]+'/tracker/SparseTrack_main')

from tracker.sparse_tracker import SparseTracker
from tools import track

def HHB_trackerload(kargs):
    args = track.make_parser().parse_args()
    args.track_thresh = 0.5
    args.track_buffer = 20
    args.down_scale = 2  
    args.layers = 2
    args.depth_levels=3
    args.depth_levels_low=0
    args.confirm_thresh=0.7
    
    return SparseTracker(args)