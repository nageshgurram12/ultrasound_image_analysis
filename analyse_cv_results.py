#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:05:52 2020

@author: nageswara
"""


from path import SYMBOLS
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file =  os.path.join(SYMBOLS.DATA_PATH, "analyse.txt")
    
    results = []
    files = []
    # each line has image file name, actual, predicted
    with open(file, "r") as fr:
        
        header = fr.readline()
        for line in fr.readlines():
            image_res = line.split(",")
            files.append(image_res[0])
            results.append([float(image_res[1]), float(image_res[2])])
    
    results = np.array(results)
    
    n = len(results)
    # std dev
    diff = (results[:, 0] - results[:, 1])
    std_dev = np.std(diff)


    plt.figure(0)
    plt.hist(diff, bins=(-1.5, -0.5, 0.5, 2.5))
    plt.xlabel("Errors"); plt.ylabel("Frequency")
    
    plt.figure(1)
    plt.scatter(range(n), diff, c='r')
    plt.plot(range(n), [0]*n, color='black', label='0 error')
    

    plt.plot(range(n), [-std_dev]*n, '--g', label='-1 std. dev')
    plt.plot(range(n), [std_dev]*n, '--b', label='+1 std.dev')
    plt.xlabel("Images Index"); plt.ylabel("Error"); plt.ylim(-1.5,3)
    plt.legend(loc="best")
    
    # file with more than +2 std. dev error
    large_err_files = {}
    err_files = {}
    
    for i in range(n):
        res = results[i]
        if np.abs(res[0]-res[1]) > 2*std_dev:
            large_err_files[files[i]] = np.abs(res[0]-res[1])
          
        err_files[files[i]] = np.abs(res[0]-res[1])
        
    large_err_files = sorted(large_err_files.items(), key = lambda x: x[1], \
                             reverse=True)
    err_files = sorted(err_files.items(), key= lambda x: x[1])[:3]
    #err_files = dict(list(err_files.items())[:3])
    print(large_err_files)
    print(err_files)
    
    # get r2 and mad
    mad = np.mean(np.abs(diff))
    sse = np.sum(np.square(diff))
    actual_mean = np.mean(results[:, 0])
    ssto = np.sum(np.square(results[:, 0] - actual_mean))
    R2 = 1-(sse/ssto)
    
    rmse = np.sqrt(sse/n)
    print(" Mean Absolute Deviation: " + str(mad))
    print("R2 : " + str(R2))
    print("RMSE :" + str(rmse))
    print(" Mean: " + str(np.mean(diff)) + " Std. Dev :" + str(std_dev))
    print(" Mean of Predictions: " + str(np.mean(results[:, 1])) + \
          " Std. Dev :" + str(np.std(results[:, 1])))
    
    print(" Mean of Actuals: " + str(np.mean(results[:, 0])) + \
          " Std. Dev :" + str(np.std(results[:, 0])))
    
    
    # plot actual vs error
    act_vs_errors = []
    for i in range(n):
        res = results[i]
        act_vs_errors.append([res[0], np.abs(res[0] - res[1])])
        
    act_vs_errors = np.array(act_vs_errors)
    plt.figure(2)
    plt.scatter(act_vs_errors[:, 0], act_vs_errors[:, 1])
    plt.xlabel("Actual Diameters"); plt.ylabel("Errors")