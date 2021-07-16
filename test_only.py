from utils import *
import argparse
import os
import imageio
import csv
from detectOJ import *
import shutil
from enhancement import *
from subprocess import call
import model
from importlib import import_module
darknet = import_module('detector.darknet')
darknet.set_gpu(1)

parser = argparse.ArgumentParser()
parser.add_argument('--inputDir', type=str, required=True)
parser.add_argument('--outputDir', type=str, required=True)

## Detection arguments
parser.add_argument("--deConf", type=str, required=True, help='path to detection net config')
parser.add_argument("--deWeight", type=str, required=True, help='path to detection model pretrained weight')
parser.add_argument("--deMetaConf", type=str,required=True, help='path to detection meta data config')
parser.add_argument("--deThresh", type=float, required=True, help='detection Threshold')

## Enhancement arguments
## BG arguments
parser.add_argument('--bgModelName', type=str, required=True)
parser.add_argument('--bgWeight', type=str, required=True)
parser.add_argument('--bgQP', type=int, required=True)
parser.add_argument("--deciThresh", type=float, required=True)

## OJ arguments
parser.add_argument('--ojModelName', type=str, required=True)
parser.add_argument('--ojWeight', type=str, required=True)
parser.add_argument('--ojQP', type=int, required=True)
parser.add_argument('--nNeibs', default=5, type=int)

## Env arguments
parser.add_argument("--width", type=int, default=1920, help='width')
parser.add_argument("--height", type=int, default=1080, help='height')
parser.add_argument("--validWidth", type=int, default=20, help='width')
parser.add_argument("--validHeight", type=int, default=10, help='height')

parser.add_argument('--format', type=str, required=False, default='.png')
parser.add_argument('--nChannels', type=int, default=3)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--nGPUs', type=int, default=1)
parser.add_argument('--cudaDevice', type=int, default=3)
parser.add_argument('--ojDepend', action='store_true')
parser.add_argument('--cpu', action='store_true')

opt = parser.parse_args()





if __name__ == '__main__':

    device = torch.device('cpu' if opt.cpu else 'cuda:'+ str(opt.cudaDevice))

    bgModel = model.Model(opt, opt.bgModelName, opt.bgWeight).to(device)
    ojModel = model.Model(opt, opt.ojModelName, opt.ojWeight).to(device)

    seqName = os.path.basename(os.path.normpath(opt.inputDir))
    createFolder(opt.outputDir, seqName, opt.ojQP, opt.bgQP)
    frameDirs = sorted(glob.glob(os.path.join(opt.inputDir, '*.png')))

    tempFol = os.path.join(opt.outputDir, seqName,'QP' +  str(opt.ojQP),'QP' +  str(opt.bgQP), 'Temp' )
    coordFol = os.path.join(opt.outputDir, seqName,'QP' +  str(opt.ojQP),'QP' +  str(opt.bgQP), 'Coord' )
    #yuvFol = os.path.join(opt.outputDir, seqName,'QP' + str(opt.ojQP),'QP' +  str(opt.bgQP), 'YUV' )
    compressedFol = os.path.join(opt.outputDir, seqName,'QP' + str(opt.ojQP), 'QP' +  str(opt.bgQP), 'Compressed' )
    decompressedFol = os.path.join(opt.outputDir, seqName, 'QP' + str(opt.ojQP),'QP' +  str(opt.bgQP), 'Decompressed' )

    tempBUDir = os.path.join(tempFol, 'BU.png')
    tempBUYUV = os.path.join(tempFol, 'yuvBU.yuv')
    tempBUNEDir = os.path.join(tempFol, 'compressedBUNE.png')

    tempRUDir = os.path.join(tempFol, 'RU.png')
    tempRUYUV = os.path.join(tempFol, 'yuvRU.yuv')
    tempRUNEDir = os.path.join(tempFol, 'compressedRUNE.png')

    trackingFile = tempFol = os.path.join(opt.outputDir, seqName,'QP' +  str(opt.ojQP),'QP' +  str(opt.bgQP), 'tracking.csv' )


    configPath = opt.deConf
    weightPath= opt.deWeight
    metaPath= opt.deMetaConf

    netMain = None
    metaMain = None

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")

    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))



    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    tracking = []
    BU = imageio.imread(frameDirs[0])
    flagBU = False
    xminBU,xmaxBU,yminBU,ymaxBU = 0,0,0,0
    #Rest
    for i,frameDir in enumerate(frameDirs):
        frame = imageio.imread(frameDir)
        idx = int(os.path.basename(frameDir).split('_')[-1].replace('.png',''))
        compressFile = os.path.join(compressedFol, 'frame_{:06d}'.format(idx) + '.hevc')
        outFrame = os.path.join(decompressedFol, 'frame_{:06d}'.format(idx) + '.png')
        recFile = os.path.join(coordFol, 'frame_{:06d}'.format(idx) + '.csv')

        print("Starting the YOLO Detection...")
        flag,xmin,xmax,ymin,ymax = YOLO(netMain, metaMain,darknet_image, frame,thresh = opt.deThresh, w = opt.width, h = opt.height)



        if (xmax-xmin) < opt.validWidth or (ymax-ymin) < opt.validHeight:
            flag = False



        bFlag = False
        bxmin,bxmax,bymin,bymax = 0,0,0,0
        if flag:
            bFlag = True
            if flagBU:
                bxmin,bxmax,bymin,bymax = calNewCor((xminBU,xmaxBU,yminBU,ymaxBU),(xmin,xmax,ymin,ymax),opt)
            else:
                bxmin,bxmax,bymin,bymax = xmin,xmax,ymin,ymax
        else:
            if flagBU:
                bFlag = True
                bxmin,bxmax,bymin,bymax = xminBU,xmaxBU,yminBU,ymaxBU
            else:
                bFlag = False



        if (i==0):
            #BU
            BU = frame
            #shutil.copyfile(BU, tempBUDir)

            tracking.append([idx,'1'])

            flagBU,xminBU,xmaxBU,yminBU,ymaxBU= flag,xmin,xmax,ymin,ymax

            WAH = str(opt.width) + 'x' + str(opt.height)

            if os.path.exists(tempBUYUV):
                os.remove(tempBUYUV)
                os.remove(tempBUNEDir)

            ## Compressed
            # convert PNG -> YUV
            call(['ffmpeg -y -i '+frameDir+' -pix_fmt yuv420p -s '+ WAH + ' ' + tempBUYUV],shell=True)
            # call X265 ->  .hevc
            call(['x265 --input-res '+ WAH +' --fps 1 --input '+tempBUYUV+' --preset 0 --qp '+ str(opt.bgQP) +' --profile main-intra --output '+ compressFile],shell=True)
            ## Decompressed
            #cal ffmpeg -> noEnhance
            call(['ffmpeg -y -i '+ compressFile + ' -f image2 -vsync 0 ' + tempBUNEDir],shell=True)

            if flagBU:
                ## Backgound Enhancement
                print("Starting the BU Enhancement")
                BUE = enhanceBG(tempBUNEDir, (xminBU,xmaxBU,yminBU,ymaxBU), bgModel)
            else:
                BUE = imageio.imread(tempBUNEDir)

            imageio.imwrite(outFrame, BUE)
            print("End the BU Out")

        else:
            #RU
            tracking.append([idx,'0'])
            flagRU,xminRU,xmaxRU,yminRU,ymaxRU = flag,xmin,xmax,ymin,ymax

            if os.path.exists(tempRUYUV):
                os.remove(tempRUYUV)

            if not flagRU:
                # No OJ, directly compressed full, no enhanced??? Don't we need to send?
                WAH = str(opt.width) + 'x' + str(opt.height)
                ## Compressed
                # convert PNG -> YUV
                call(['ffmpeg -y -i '+frameDir+' -pix_fmt yuv420p -s '+ WAH + ' ' + tempRUYUV],shell=True)
                # call X265 ->  .hevc
                call(['x265 --input-res '+ WAH +' --fps 1 --input '+tempRUYUV+' --preset 0 --qp '+ str(opt.bgQP) +' --profile main-intra --output '+ compressFile],shell=True)
                ## Decompressed
                #cal ffmpeg -> noEnhance
                call(['ffmpeg -y -i '+ compressFile + ' -f image2 -vsync 0 ' + outFrame],shell=True)
            else:
                if not flagBU:
                    xminTrue,xmaxTrue,yminTrue,ymaxTrue = [xminRU,xmaxRU,yminRU,ymaxRU]
                else:
                    xminTrue,xmaxTrue,yminTrue,ymaxTrue = bxmin,bxmax,bymin,bymax

                OJ = imageio.imread(frameDir)
                OJ = OJ[yminTrue:ymaxTrue+1,xminTrue:xmaxTrue+1,:]
                imageio.imwrite(tempRUDir, OJ)
                WAH = str(OJ.shape[1]) + 'x' + str(OJ.shape[0])

                if os.path.exists(tempRUNEDir):
                    os.remove(tempRUNEDir)
                ## Compressed
                # convert PNG -> YUV
                call(['ffmpeg -i '+tempRUDir+' -pix_fmt yuv420p -s '+ WAH + ' ' + tempRUYUV],shell=True)
                # call X265 ->  .hevc
                call(['x265 --input-res '+ WAH +' --fps 1 --input '+tempRUYUV+' --preset 0 --qp '+ str(opt.ojQP) +' --profile main-intra --output '+ compressFile],shell=True)
                ## Decompressed
                #cal ffmpeg -> noEnhance
                call(['ffmpeg -i '+ compressFile + ' -f image2 -vsync 0 ' + tempRUNEDir],shell=True)

                print("Starting the RU Enhancement")
                tempRUNE = imageio.imread(tempRUNEDir)
                RU = enhanceOJ(opt, tempRUNE, (xminTrue,xmaxTrue,yminTrue,ymaxTrue), BUE, ojModel,opt.nNeibs)

                imageio.imwrite(outFrame, RU)
                print("End the RU Out")

        with open(recFile,mode='w') as target:
            writer = csv.writer(target)
            record = [flag,xmin,xmax,ymin,ymax]
            writer.writerow(record)
            target.close()


    with open(trackingFile,mode='w') as target:
        writer = csv.writer(target)
        nBU = 0
        for row in tracking:
            if row[-1] == '1': nBU+=1
            record = row
            writer.writerow(record)
        writer.writerow(['BU', nBU])
        writer.writerow(['RU', len(tracking) - nBU])
        writer.writerow(['BURatio', nBU/len(tracking)])
        writer.writerow(['RURatio', (len(tracking) - nBU)/len(tracking)])
        target.close()
