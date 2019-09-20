#!/usr/bin/python

# given input train log file, extract loss function on validation and training. Feed into python script for plotting

import sys, getopt, os
import matplotlib.pyplot as plt
from subprocess import PIPE, Popen
import numpy as np


def movingAverageConv(a, window_size=1) :
    if not a : return a
    window = np.ones(int(window_size))
    result = np.convolve(a, window, 'full')[ : len(a)] # Convolve full returns array of shape ( M + N - 1 ).
    slotsWithIncompleteConvolution = min(len(a), window_size-1)
    result[slotsWithIncompleteConvolution:] = result[slotsWithIncompleteConvolution:]/float(window_size)
    if slotsWithIncompleteConvolution > 1 :
        divisorArr = np.asarray(range(1, slotsWithIncompleteConvolution+1, 1), dtype=float)
        result[ : slotsWithIncompleteConvolution] = result[ : slotsWithIncompleteConvolution] / divisorArr
    return result

def makeFiles(argv):
    file = ''
    w = 1
    save = False
    try:
		opts, args = getopt.getopt(argv,"hsf:m:",["file=", "movingAverage="])
    except getopt.GetoptError:
		print('plotLossFunctionKeras.py -f <train log full file address: /home/...> -m <moving average>')
		sys.exit(2)
    for opt, arg in opts:
		if opt == '-h':
			print('plotLossFunctionKeras.py -f <train log full file address : /home/...> -m <moving average>')
			sys.exit()
		elif opt in ("-f","--file"):
			file = str(arg)
		elif opt in ("-m","--movingAverage"):
			w = int(arg)   
		elif opt in ("-s"):
			save = True   

    print(w)

    bashCommand_getTrain = "grep 'Train cost and metrics' " +  file + " | awk '{print $5}' | grep -o '[0-9].*' | sed 's/,//' "
    bashCommand_getDICE1Train = "grep 'Train cost and metrics' " + file +  " | awk '{print $6}' | grep -o '[0-9].*' | sed 's/]//' | sed 's/,//' "
    bashCommand_getDICE2Train = "grep 'Train cost and metrics' " + file +  " | awk '{print $7}' | grep -o '[0-9].*' | sed 's/]//'| sed 's/,//' "



    bashCommand_getVal = "grep 'Validation cost and accuracy' " +  file + " | awk '{print $5}' | grep -o '[0-9].*' | sed 's/,//' "
    bashCommand_getDICE1Val = "grep 'Validation cost and accuracy' " + file +  " | awk '{print $6}' | grep -o '[0-9].*' | sed 's/]//' | sed 's/,//'"
    bashCommand_getDICE2Val = "grep 'Validation cost and accuracy' " + file +  " | awk '{print $7}' | grep -o '[0-9].*' | sed 's/]//' | sed 's/,//' "


    bashCommand_getDSC = "grep 'Overall' " + file +  " | awk '{print $3}' "

    p = Popen(bashCommand_getTrain, stdout=PIPE, shell=True)
    output = p.communicate()
    train = output[0].split()

    p = Popen(bashCommand_getDICE1Train, stdout=PIPE, shell=True)
    output = p.communicate()
    Tdice1 = output[0].split()
    	
    p = Popen(bashCommand_getDICE2Train, stdout=PIPE, shell=True)
    output = p.communicate()
    Tdice2 = output[0].split()

    p = Popen(bashCommand_getVal, stdout=PIPE, shell=True)
    output = p.communicate()
    Val = output[0].split()

    p = Popen(bashCommand_getDICE1Val, stdout=PIPE, shell=True)
    output = p.communicate()
    Valdice1 = output[0].split()

    p = Popen(bashCommand_getDICE2Val, stdout=PIPE, shell=True)
    output = p.communicate()
    Valdice2 = output[0].split()

    p = Popen(bashCommand_getDSC, stdout=PIPE, shell=True)
    output = p.communicate()
    DSC = output[0].split()


    for i in range(0,len(train)-1):
        train[i] = float(train[i])
    train = train[:-1]

    for i in range(0,len(Tdice1)-1):
        Tdice1[i] = float(Tdice1[i])
    Tdice1 = Tdice1[:-1]
    
    for i in range(0, len(Tdice2)-1):
        Tdice2[i] = float(Tdice2[i])
    Tdice2 = Tdice2[:-1]

    for i in range(0, len(Val)-1):
    	Val[i] = float(Val[i])
    Val = Val[:-1]

    for i in range(0, len(Valdice1)):
    	Valdice1[i] = float(Valdice1[i])

    for i in range(0, len(Valdice2)):
    	Valdice2[i] = float(Valdice2[i])

    for i in range(0, len(DSC)):
    	DSC[i] = float(DSC[i])



    train = movingAverageConv(train, window_size = w)
    Tdice1 = movingAverageConv(Tdice1, window_size = w)
    Tdice2 = movingAverageConv(Tdice2, window_size = w)
    
    #Val = movingAverageConv(Val, window_size = w)
    #Valdice1 = movingAverageConv(Valdice1, window_size = w)
    #Valdice2 = movingAverageConv(Valdice2, window_size = w)



    plt.clf()

    plt.subplot(511)
    plt.plot(range(len(train)),train,'k-')
    #plt.xlabel('weight updates')
    plt.title('Training Data - moving average {}'.format(w),)
    plt.axis('tight')
    plt.legend(('Training Loss',))

    plt.subplot(512)
    plt.plot(range(0,len(Val)),Val,'k-')
    plt.legend(('Validation Loss',))

    
    plt.subplot(513)
    plt.plot(range(len(Tdice1)),Tdice1,'r-')
    plt.plot(range(len(Tdice2)),Tdice2,'b-')
    plt.legend(('Training Dice0', 'Training Dice1'))

    plt.subplot(514)
    plt.plot(range(0,len(Valdice1)),Valdice1,'r-')
    plt.plot(range(0,len(Valdice2)),Valdice2,'b-')
    plt.legend(('Val Dice0', 'Val Dice1'))

    plt.subplot(515)
    plt.plot(range(len(DSC)),DSC,'o-')
    plt.legend(('Full Segmentation Dice',))
    
    if save:

        out = '/'.join(file.split('/')[:-1])
	print('saved in {}'.format(out))
    	plt.savefig(out+'/Training_session.png', bbox_inches='tight')
	
	plt.clf()

    else:

        plt.show()
    
    
if __name__ == "__main__":
	makeFiles(sys.argv[1:])
    
    
    
