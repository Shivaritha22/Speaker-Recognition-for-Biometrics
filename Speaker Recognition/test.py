from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from mel_coefficients import mfcc


from train import training

nSpeaker = 8
nfiltbank = 22

(codebooks_mfcc) = training(nfiltbank)
directoryz = 'C:/Users/hp/Desktop/review 2 final/zero/test'
directoryh='C:/Users/sachi/Documents/college/sem5/DSP/review 2 final/hello/test_hello'
directory1 = 'C:/Users/hp/Desktop/review 2 final/zero/speaker'
directory2='C:/Users/sachi/Documents/college/sem5/DSP/review 2 final/hello/speaker'


def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k, :, :])
        dist = np.sum(np.min(D, axis=1)) / (np.shape(D)[0])
        if dist < distmin:
            distmin = dist
            speaker = k
    return speaker

def z1():
    nCorrect_MFCC = 0
    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'

        print('Now speaker ', str(i + 1), 'features are being tested')
        (fs, s) = read(directoryz + fname)

        mel_coefs = mfcc(s, fs, nfiltbank)

        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)


        print('Speaker', (i + 1), ' in test matches with speaker ', (sp_mfcc + 1), 'in train for training with MFCC')


        if i == sp_mfcc:
            nCorrect_MFCC += 1
    percentageCorrect_MFCC = (nCorrect_MFCC / nSpeaker) * 100
    print('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')

def z2():
    nCorrect_MFCC = 0
    fname = input("enter wav file")
    for i in range(nSpeaker):

        print('Now speaker ', str(i + 1), 'features are being tested')
        (fs, s) = read(directory1 + fname)

        mel_coefs = mfcc(s, fs, nfiltbank)

        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)

        # print('Speaker', (i + 1), ' in test matches with speaker ', (sp_mfcc + 1), 'in train for training with MFCC')

        if i == sp_mfcc:
            nCorrect_MFCC += 1
    # percentageCorrect_MFCC = (nCorrect_MFCC / nSpeaker) * 100
    # print('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')
    names = {1: "gaurav", 2: "yuvan", 3: "khamalesh", 4: "lekha", 5: "lalitha", 6:"shivaritha", 7: "surya", 8: "Senthil" }
    print('Identified Speaker:', names[sp_mfcc + 1])




inp=input("enter choice:\n1.show percentage(p)\n2.identify speaker(s)")
if inp=="1":
    z1()
elif inp=="2":
    z2()
