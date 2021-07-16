import os
import glob
import imageio
import numpy as np
from skimage.metrics import structural_similarity as ssim

def createFolder(outRoot, seqName, ojQP, bgQP):
    seqFol = os.path.join(outRoot, seqName,'QP' + str(ojQP), 'QP' + str(bgQP))
    if not os.path.exists(seqFol):
        os.makedirs(seqFol,exist_ok=True)

    folList = ['Temp','Coord','Compressed','Decompressed']
    for fol in folList:
        os.makedirs(os.path.join(seqFol,fol),exist_ok=True)

def calDistance(prevBU, frame):
    prevBU = np.array(prevBU)
    #prevBU = np.expand_dims(prevBU,axis =0)
    frame = np.array(frame)
    #frame = np.expand_dims(frame, axis =0)
    #distance =  1 - ssim(prevBU,frame,multichannel = True)
    distance = 1 - ssim( prevBU, frame, multichannel = True )
    return distance

def makeDecision( prevBU, frame, cor,bFlag = True,thresh = 0.1, order = 10):
    if order == 0:
        return True

    #prevBU =imageio.imread(prevBU)
    #frame =imageio.imread(frame)
    BU =prevBU
    cur = frame
    if bFlag:
        xmin, xmax, ymin, ymax = cor
        BU[int(ymin):int(ymax)+1,int(xmin):int(xmax)+1,:] = 0
        cur[int(ymin):int(ymax)+1,int(xmin):int(xmax)+1,:] = 0

    distance = calDistance(BU, cur)
    if distance > thresh: return True
    return False

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range).squeeze()

def calNewCor(corBU, corRU, opt):
    xminRU,xmaxRU,yminRU,ymaxRU = corRU
    xminBU,xmaxBU,yminBU,ymaxBU = corBU
    xmin = min(xminRU,xminBU)
    ymin = min(yminRU,yminBU)
    xmax = max(xmaxRU,xmaxBU)
    ymax = max(ymaxRU,ymaxBU)

    if not ((xmax-xmin +1)%2 ==0):
        if not xmax == (opt.width-1):
            xmax= xmax+1
        elif not (xmin == 0):
            xmin = xmin - 1
        else:
            xmax = xmax - 1

    if not ((ymax-ymin +1)%2 == 0):
        if not ymax == (opt.height-1):
            ymax= ymax+1
        elif not ymin == 0:
            ymin = ymin - 1
        else:
            ymax = ymax - 1

    return xmin, xmax, ymin, ymax
