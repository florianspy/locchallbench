import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy
import evaluate_rpe as tr
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys
import os
from pathlib import Path
import math
import mplcursors
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
_EPS = numpy.finfo(float).eps * 4.0
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_NEXT_AXIS = [1, 2, 0, 1]


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True
    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def evaluate(eval_path, gt_path):
    print("File to be evaluated:{}".format(eval_path))
    axis = numpy.array([[0.0, 4.5, 3.2, 9.5], [-5, 10, -10, 20], [-1, 16, -7, 20]])
    setnum=0
    output_plot_file_name = "out.pdf"    
    texfile = "latex.tex"
    try:
        os.remove(texfile)
        print("File Removed!")
    except:
         print("File not there")
    def remove_old_output_file():
        try:
            os.remove(output_plot_file_name)
        except:
            print("{} could not be deleted as it does not exist.".format(output_plot_file_name))


    def read_file(filepath_to_read):
        with open(filepath_to_read) as gt_in:
            return (np.fromstring(gt_in.read(), sep="   ").reshape(-1, 8))


    def plot_data(data, str_color):
        plt.plot(data[:, 1], data[:, 2], str_color, linewidth=2)
        starting_point = (data[1, 1], data[1, 2])
        end_point = (data[len(data) - 1, 1], data[len(data) - 1, 2])
        plt.plot(starting_point[0], starting_point[1], 'kx', markersize=7, mew=2)
        plt.plot(end_point[0], end_point[1], 'k+', markersize=10, mew=2)

    def getmaxmin(eval_data, maxmin):
        if eval_data[1] < maxmin[0]:
            maxmin[0]=eval_data[1]
        if eval_data[1] > maxmin[1]:
            maxmin[1]=eval_data[1]
        if eval_data[2] < maxmin[2]:
            maxmin[2]=eval_data[2]
        if eval_data[2] > maxmin[3]:
            maxmin[3]=eval_data[2]

    def getpathlength(data):
        path_length =0       	
        prev_x=0
        prev_y=0
        for i in range(0, np.size(data, 0) - 1):
            cur_x=data[i,1]
            cur_y=data[i,2] 
            if i != 0:                                           
               path_length = path_length +math.sqrt(math.pow(cur_x-prev_x,2)+math.pow(cur_y-prev_y,2))
            prev_x=cur_x
            prev_y=cur_y
        return path_length      	
	
    '''Prepare the plot'''
    f = plt.figure()
    plt.grid()
    
    plt.axis('equal') 
    plt.gca().set_aspect('equal', adjustable='box')
    f.tight_layout()   
    '''Begin Main'''
    remove_old_output_file()
    gt_data = read_file(gt_path)
    eval_data = read_file(eval_path)  
    maxmin=[eval_data[0,1],eval_data[0,1],eval_data[0,2] ,eval_data[0,2]]    
    plot_data(gt_data, 'black')
    plot_data(eval_data, 'orange')
    head, tail = os.path.split(eval_path)
    first=0
    end=0
    #https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
    for index,value in enumerate(tail):
        if tail[index:index+(len("-"))] == "-":
            if first == 0:
                first=index
            else:
                end=index
    plt.ylabel('y-Position in m',fontsize=28)
    plt.xlabel('x-Position in m',fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    orange_patch = Line2D([], [], color='orange', lw=2,label=tail[first+1:end],marker='_',linestyle='None', markersize=15,mew=2) 
    black_patch = Line2D([], [], color='black', lw=2,label='Ground Truth',marker='_',linestyle='None', markersize=15,mew=2) 
    start = Line2D([], [], color='black', marker='x',label='Start',linestyle='None', markersize=10,mew=2) 
    goal = Line2D([], [], color='black', marker='+',label='End',linestyle='None', markersize=10,mew=2)
    patchList = []
    patchList.append(orange_patch)
    patchList.append(black_patch)
    patchList.append(start)
    patchList.append(goal)
    first_legend=plt.legend(handles=patchList,loc='upper left',fancybox=True, framealpha=0.5, markerscale=1,handletextpad=0.,bbox_to_anchor=(-0.02, 1.02),frameon=False,fontsize=24)  #    
    skip = 1
    # http://akuederle.com/create-numpy-array-with-for-loop
    error_array = np.array([])
    error_ang_array = np.array([])
    klast = 0
    gtlength=getpathlength(gt_data)
    evallength=getpathlength(eval_data)
    totaldata=0
    prevgt=[0,0,0]
    firsteval=1
    discon=0.0
    disconcount=0
    for i in range(0, np.size(eval_data, 0) - 1): 
        getmaxmin(eval_data[i],maxmin)            
        if eval_data[i, 0] < gt_data[0, 0]:
            print("no gt data available for this timestamp")
        else:
            if eval_data[i, 0] > gt_data[np.size(gt_data, 0) - 1, 0]:
                print("no more gt data available")
                break
            for k in range(klast, np.size(gt_data, 0) - 1):
                skip = 1
                if eval_data[i, 0] >= gt_data[k, 0] and eval_data[i, 0] < gt_data[k + 1, 0]:
                    #print( "match at",eval_data[i,0],gt_data[k,0],i,k,klast,np.size(gt_data, 0))
                    klast = k
                    skip = 0
                    break
            if skip == 0:
                a = (eval_data[i, 0] - gt_data[k, 0]) / (gt_data[k + 1, 0] - gt_data[k, 0])
                x_gt = a * gt_data[k + 1, 1] + (1 - a) * gt_data[k, 1]
                y_gt = a * gt_data[k + 1, 2] + (1 - a) * gt_data[k, 2]
                z_gt = 0
                #z_gt = a * gt_data[k + 1, 3] + (1 - a) * gt_data[k, 3]		
                e = np.sqrt((eval_data[i, 1] - x_gt) ** 2 + (eval_data[i, 2] - y_gt) ** 2)
                #e = np.sqrt((eval_data[i, 1] - x_gt) ** 2 + (eval_data[i, 2] - y_gt) ** 2+ (eval_data[i, 3] - z_gt) ** 2)
                error_array = np.append(error_array, e)
                if firsteval == 1:
                   firsteval = 0
                else:
                   vecgt=[x_gt-prevgt[0],y_gt-prevgt[1]]
                   #vecgt=[x_gt-prevgt[0],y_gt-prevgt[1],z_gt-prevgt[2]]
                   #the jump occurred between i-1 and i so if you want to look at it inspect the data at i-1
                   veceval=[eval_data[i, 1]-eval_data[i-1, 1],eval_data[i, 2]-eval_data[i-1, 2]]
                   #veceval=[eval_data[i, 1]-eval_data[i-1, 1],eval_data[i, 2]-eval_data[i-1, 2],eval_data[i, 3]-eval_data[i-1, 3]]
                   lenvecgt=math.sqrt(math.pow(vecgt[0],2)+math.pow(vecgt[1],2))
                   #lenvecgt=math.sqrt(math.pow(vecgt[0],2)+math.pow(vecgt[1],2)+math.pow(vecgt[2],2))
                   #we cannot just subtract veceval from vecgt because if there is a rotation in the evaluation data the vectors' will point in different directions,
                   #with the calculate length then being too large. We therefor directly compare the lenght of the two vectors.                   
                   lengveceval=math.sqrt(math.pow(veceval[0],2)+math.pow(veceval[1],2))
                   #lengveceval=math.sqrt(math.pow(veceval[0],2)+math.pow(veceval[1],2)+math.pow(veceval[2],2))
                   if lenvecgt > 0.0001 :
                      discon=discon+math.pow(1-lengveceval/lenvecgt,2)
                      disconcount=disconcount+1                
                prevgt=[x_gt,y_gt,z_gt]
                #angle at k+1 and angle at current k
                angleskp1 = euler_from_quaternion([gt_data[k + 1, 4], gt_data[k + 1,5],gt_data[k + 1,6], gt_data[k + 1, 7]])
                anglesk = euler_from_quaternion([gt_data[k, 4], gt_data[k,5],gt_data[k,6], gt_data[k, 7]])
                if anglesk[2]>0 and angleskp1[2]<0 :	
                      angleskp1=angleskp1[2]+math.pi*2
                      anglez_gt=a*angleskp1+(1-a)*anglesk[2]
                      if anglez_gt > math.pi:
                      	anglez_gt=anglez_gt-math.pi*2
                else:                
                      anglez_gt=a*angleskp1[2]+(1-a)*anglesk[2]
                angleval= euler_from_quaternion([eval_data[i, 4], eval_data[i,5],eval_data[i,6], eval_data[i, 7]])
                eang=angleval[2]-anglez_gt
                error_ang_array = np.append(error_ang_array,eang)
    if len(error_array) == 0:
        print("No Data with matching timestamps")
        return
    rmse = np.sqrt(np.dot(error_array, error_array) / len(error_array))
    mean = np.mean(error_array)
    meanang = np.mean(error_ang_array)
    median = np.median(error_array)
    std = np.std(error_array)
    max = np.max(error_array)
    maxang=np.max(error_ang_array)    
    meanang = np.mean(error_ang_array)
    medianang = np.median(error_ang_array)
    stdang = np.std(error_ang_array)
    traj_gt = tr.read_trajectory(gt_path)
    traj_est = tr.read_trajectory(eval_path)
    result = tr.evaluate_trajectory(traj_gt, traj_est, 10000, True, 1.00, "s", 0.00, 1.00)
    trans_error = numpy.array(result)[:, 4]
    rot_error = numpy.array(result)[:, 5]
    rmset = numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))
    rmser = numpy.sqrt(numpy.dot(rot_error, rot_error) / len(rot_error))
    relpath=evallength/gtlength
    discoty=math.sqrt(discon)/disconcount
    with open(texfile, "a") as param:
        writing = tail[first+1:end] + " & {:.3f}".format(rmse) + " & {:.3f}".format(rmset)+ " & {:.3f}".format(rmser) + " & {:.3f}".format(max)+ " & {:.3f}".format(maxang) + " & {:.3f}".format(mean)+ " & {:.3f}".format(meanang) + " & {:.3f}".format(median)+ " & {:.3f}".format(medianang) + " & {:.3f}".format(std)+ " & {:.3f}".format(stdang)+ " & {:.3f}".format(relpath*100.0)+ " & {:.3f}".format(discoty)
        param.write(writing)
    with open(texfile, "r+") as param:
        content = param.read()
        param.seek(0, 0)
        writing = "Algorithm & RMSE(at) & RMSE(rt) & RMSE(rr) & Maxt & Maxr & Meant & Meanr & Mediant & Medianr & Stdt & Stdr & Pathratio & Discontinuity "   
        writing = writing + " \\\ \hline \n"             
        param.write(writing + content)
    print("Algorithm & RMSE(at) & RMSE(rt) & RMSE(rr) & Maxt & Maxr & Meant & Meanr & Mediant & Medianr & Stdt & Stdr & Pathratio & Discontinuity ")
    print(rmse,rmset,rmser,max,maxang, mean, meanang, median,medianang, std, stdang,relpath*100.0,discoty)
    '''crs=mplcursors.cursor(hover=True)
    crs.connect("add", lambda sel: sel.annotation.set_text('Point {},{}'.format(sel.target[0], sel.target[1])))
    plt.show()'''	
    plt.xlim(maxmin[0]-1, maxmin[1]+1)
    plt.ylim(maxmin[2]-1, maxmin[3]+1)
    '''Save Plot'''
    DefaultSize = f.get_size_inches()
    f.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
    f.savefig(output_plot_file_name, bbox_inches='tight')
    output_plot_file_name = tail[first+1:end]+".pdf"
    f.savefig(output_plot_file_name, bbox_inches='tight')
if __name__ == "__main__":
    evaluate("0-visual_odom_orb.txt", "0-gt.txt")
