# [x coord of center of object, y coord, width, height, one-hot label vector]
# Probably as a numpy array vector
# Should be similar to what yolo outputs, so check how that works
# pixel means whether the representation is in the form of pixels (integers, True) or decimals (False)
import numpy as np
def iou(expected_info, calculated_info, pixel = True, has_class = True, class_first=False):
    '''
    exp_width = expected_info[0][2]
    exp_height = expected_info[0][3]
    exp_cx = expected_info[0][0]
    exp_cy = expected_info[0][1]
    exp_tl = (exp_cx-exp_width//2, exp_cy-exp_height//2)
    exp_br = (exp_cx+exp_width//2, exp_cy+exp_height//2)
    exp_area = exp_width*exp_height

    exp_class = np.array(expected_info[1])
    '''
    x, y, w, h = [0, 1, 2, 3]
    if has_class:
        c, x, y, w, h = [4, 0, 1, 2, 3]
    if class_first:
        c, x, y, w, h = [0, 1, 2, 3, 4]

    if has_class:
        exp_class = np.array(expected_info[c])
    exp_width = expected_info[w]
    exp_height = expected_info[h]
    exp_cx = expected_info[x]
    exp_cy = expected_info[y]
    if pixel:
        exp_tl = (exp_cx - exp_width // 2, exp_cy - exp_height // 2)
        exp_br = (exp_cx + exp_width // 2, exp_cy + exp_height // 2)
    else:
        exp_tl = (exp_cx - exp_width / 2, exp_cy - exp_height / 2)
        exp_br = (exp_cx + exp_width / 2, exp_cy + exp_height / 2)
    exp_area = exp_width * exp_height

    if has_class:
        calc_class = np.array(calculated_info[c])
    calc_width = calculated_info[w]
    calc_height = calculated_info[h]
    calc_cx = calculated_info[x]
    calc_cy = calculated_info[y]
    if pixel:
        calc_tl = (calc_cx - calc_width // 2, calc_cy - calc_height // 2)
        calc_br = (calc_cx + calc_width // 2, calc_cy + calc_height // 2)
    else:
        calc_tl = (calc_cx - calc_width / 2, calc_cy - calc_height / 2)
        calc_br = (calc_cx + calc_width / 2, calc_cy + calc_height / 2)
    calc_area = calc_width*calc_height

    if has_class:
        if exp_class != calc_class:
            same_class = False
        else:
            same_class = True

    overlap_area = 0
    overlap_dim1 = min(exp_br[0], calc_br[0])-max(exp_tl[0], calc_tl[0])
    overlap_dim2 = min(exp_br[1], calc_br[1])-max(exp_tl[1], calc_tl[1])
    if overlap_dim1 > 0 and overlap_dim2 > 0:
        overlap_area = overlap_dim1*overlap_dim2

    union_area = exp_area+calc_area-overlap_area

    if has_class:
        return (overlap_area/union_area, same_class)
    else:
        return overlap_area/union_area

#print(iou([100, 200, 350,450,5], [100,200,300,400,5]))