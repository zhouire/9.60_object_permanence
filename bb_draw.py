from PIL import Image, ImageDraw, ImageFilter

def bb_draw(image, bb_info):
    draw = ImageDraw.Draw(image)
    upper_left = (bb_info[0]-bb_info[2]//2, bb_info[1]-bb_info[3]//2)
    lower_right = (bb_info[0]+bb_info[2]//2, bb_info[1]+bb_info[3]//2)
    draw.rectangle((upper_left, lower_right), outline=(255, 255, 255, 255))
    return image


#image = Image.open(r"data\pretrain\images\pretrainimg_0.jpg")
#p = bb_draw(image,(100,100, 20, 20))
#p.show()