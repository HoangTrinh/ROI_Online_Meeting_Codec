import os
from subprocess import call

#I/O
inputRoot = '/data1/hoangtm/HumanX265Test/QP2GT/QP0/meeting/'
outputRoot = '/data1/hoangtm/humanResults/BURU_GOF_meeting_only/'

#Detection params
deNetCF = 'detector/cfg/yolov4-tiny.cfg'
deMetaCF = 'detector/cfg/coco.data'
deWeight = 'detector/weight/yolov4-tiny.weights'
deThresh = 0.5


wd = 1920
height = 1080
deciThresh = 0.15
validWidth = 30
validHeight = 30
cudaDevice = 2

#Enhancement params
##BU
bgModel = 'DRRN'
bgRootWeight = '/data1/hoangtm/Pyramid-Attention-Networks/DN_RGB/experiment/normal/DRRN_QP'
##RU
ojModel = 'DRRN'
ojRootWeight = '/data1/hoangtm/Pyramid-Attention-Networks/DN_RGB/experiment/normal/DRRN_QP'
nNeibs = 30


ojQPs = [ '22','27','32','37']
#bgQPs = ['47']

seqNames = os.listdir(inputRoot)

#seqNames = ['190312_24_ParkVillaBorghese_UHD_002']

for name in seqNames:
    inputFol = os.path.join(inputRoot,name)
    for ojQP in ojQPs:
        #'''
        if ojQP in ['22','27']:
            bgQPs = [ '32','37', '42','47']
        else:
            bgQPs = ['42','47']
        #'''
        ojWeight = ojRootWeight + ojQP + '/model/model_best.pt'
        for bgQP in bgQPs:
            bgWeight = bgRootWeight + bgQP + '/model/model_best.pt'

            call(['python test_only.py --inputDir '+ inputFol +' --outputDir '+ outputRoot +' --deConf '+ deNetCF +' --deWeight '+ deWeight +' --deMetaConf '+ deMetaCF +' --deThresh '+ str(deThresh)
                +' --bgModelName '+ bgModel +' --bgWeight '+ bgWeight +' --bgQP '+ bgQP +' --deciThresh '+ str(deciThresh) +' --ojModelName '+ ojModel +' --ojWeight '+ ojWeight +' --ojQP '
                + ojQP +' --nNeibs '+ str(nNeibs)  +' --width '+ str(wd) +' --height '+ str(height) +' --validWidth '+ str(validWidth) +' --validHeight '+ str(validHeight) +' --cudaDevice '+ str(cudaDevice)  ],shell=True)
