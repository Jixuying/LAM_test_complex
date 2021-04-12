import torch
import numpy as np
from scipy import interpolate
from scipy.io import loadmat
# Num_RB_t = 2
# Num_RB_f = 6
# Num_RE_t = Num_RB_t * 7
# Num_RE_f = Num_RB_f * 12
# Num_of_pilots = 48
#
# # load datasets
# channel_model = "VehA"
# SNR = 12
# Number_of_pilots = 48
# perfect = loadmat("my_perfect_"+ 'H_40.mat')['my_perfect_H_40']
# # noisy_input = loadmat("My_noisy_" + 'H' + "_" + "SNR_" + str(SNR) + ".mat") [channel_model+"_noisy_"+ str(SNR)]
# noisy_input = loadmat("My_noisy_" + 'H' + "_" + str(SNR) + ".mat")['my_noisy_H']
#
# perfect_r = np.real(perfect)
# perfect_i = np.imag(perfect)
#
# noisy_input_r = np.real(noisy_input)
# noisy_input_i = np.imag(noisy_input)
#
# label_r = torch.from_numpy(perfect_r)
# label_i = torch.from_numpy(perfect_i)
#
# train_input_r = torch.Tensor(noisy_input_r)
# train_input_i = torch.from_numpy(noisy_input_i)
#
# torch.save(label_r,'x_r.t')
# torch.save(label_i,'x_i.t')
# torch.save(train_input_r,'input_r.t')
# torch.save(train_input_i,'input_i.t')

def interpolation(noisy_input , batch_size , Number_of_pilot , interp):
    noisy_image = torch.zeros(len(noisy_input[:, 1, 1]), 2, len(noisy_input[1, :, 1]), len(noisy_input[1, 1, :]))
    # noisy_image = np.zeros((40,72,14,2))
    noisy_input_r = np.real(noisy_input)
    noisy_input_i = np.imag(noisy_input)
    x_r = torch.from_numpy(noisy_input_r)
    x_i = torch.from_numpy(noisy_input_i)
    noisy_image[:,0,:,:] = x_r
    noisy_image[:,1,:,:] = x_i


    if (Number_of_pilot == 48):
        idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
    elif (Number_of_pilot == 16):
        idx= [4+14*(i) for i in range(1, 72,9)]+[9+14*(i) for i in range(4, 72,9)]
    elif (Number_of_pilot == 24):
        idx = [14*i for i in range(1,72,9)]+ [6+14*i for i in range(4,72,9)]+ [11+14*i for i in range(1,72,9)]
    elif (Number_of_pilot == 8):
      idx = [4+14*(i) for  i in range(5,72,18)]+[9+14*(i) for i in range(8,72,18)]
    elif (Number_of_pilot == 36):
      idx = [14*(i) for  i in range(1,72,6)]+[6+14*(i) for i in range(4,72,6)] + [11+14*i for i in range(1,72,6)]



    r = [x//14 for x in idx]
    c = [x%14 for x in idx]



    interp_noisy = np.zeros((96000,2,72,14))

    for i in range(len(noisy_input)):
        z = [noisy_image[i,0,j,k] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,0,:,:] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,0] = z_intp
        z = [noisy_image[i,1,j,k] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,1,:,:] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,1] = z_intp


    #interp_noisy = np.concatenate((interp_noisy[:,:,:,0], interp_noisy[:,:,:,1]), axis=0).reshape(80, 72, 14, 1)
    return interp_noisy

def interpolation_real(noisy , SNR , Number_of_pilot , interp):
    noisy_image = np.zeros((40,72,14,2))

    noisy_image[:,:,:,0] = np.real(noisy)
    noisy_image[:,:,:,1] = np.imag(noisy)


    if (Number_of_pilot == 48):
        idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
    elif (Number_of_pilot == 16):
        idx= [4+14*(i) for i in range(1, 72,9)]+[9+14*(i) for i in range(4, 72,9)]
    elif (Number_of_pilot == 24):
        idx = [14*i for i in range(1,72,9)]+ [6+14*i for i in range(4,72,9)]+ [11+14*i for i in range(1,72,9)]
    elif (Number_of_pilot == 8):
      idx = [4+14*(i) for  i in range(5,72,18)]+[9+14*(i) for i in range(8,72,18)]
    elif (Number_of_pilot == 36):
      idx = [14*(i) for  i in range(1,72,6)]+[6+14*(i) for i in range(4,72,6)] + [11+14*i for i in range(1,72,6)]



    r = [x//14 for x in idx]
    c = [x%14 for x in idx]



    interp_noisy = np.zeros((40,72,14,2))

    for i in range(len(noisy)):
        z = [noisy_image[i,j,k,0] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,0] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,0] = z_intp
        z = [noisy_image[i,j,k,1] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,1] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,1] = z_intp


    interp_noisy = np.concatenate((interp_noisy[:,:,:,0], interp_noisy[:,:,:,1]), axis=0).reshape(80, 72, 14, 1)
    return interp_noisy