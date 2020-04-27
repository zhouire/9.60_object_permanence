# [x coord of center of object, y coord, width, height, one-hot label vector]
# Probably as a numpy array vector
# Should be similar to what yolo outputs, so check how that works
import numpy as np
def iou(expected_info, calculated_info):
    exp_width = expected_info[0][2]
    exp_height = expected_info[0][3]
    exp_cx = expected_info[0][0]
    exp_cy = expected_info[0][1]
    exp_tl = (exp_cx-exp_width//2, exp_cy-exp_height//2)
    exp_br = (exp_cx+exp_width//2, exp_cy+exp_height//2)
    exp_area = exp_width*exp_height

    exp_class = np.array(expected_info[1])

    calc_class = np.array(calculated_info[5])
    calc_width = calculated_info[2]
    calc_height = calculated_info[3]
    calc_cx = calculated_info[0]
    calc_cy = calculated_info[1]
    calc_tl = (calc_cx-calc_width//2, calc_cy-calc_height//2)
    calc_br = (calc_cx+calc_width//2, calc_cy+calc_height//2)
    calc_area = calc_width*calc_height

    if exp_class != calc_class:
        same_class = False
    else:
        same_class = True
    
    overlap_area = (min(exp_br[0], calc_br[0])-max(exp_tl[0], calc_tl[0]))*(min(exp_br[1], calc_br[1])-max(exp_tl[1], calc_tl[1]))
    overlap_area = max(overlap_area, 0)

    union_area = exp_area+calc_area-overlap_area

    return (overlap_area/union_area, same_class)