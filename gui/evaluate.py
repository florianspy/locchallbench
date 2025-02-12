#caution timestamps can only be 15 digits long as the number after the digit is already 9 only 6 digits before the dot are allowed
#The reason for this is that python stores all the data in float variables which can only handle 15 digits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy
import evaluate_rpe as tr
import evaluate_ate as ate
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys
import os
from pathlib import Path
import math
import mplcursors

mpl.rc('font',family='Times New Roman')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
#from https://github.com/matthew-brett/transforms3d
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

#from https://github.com/matthew-brett/transforms3d
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
#from https://github.com/matthew-brett/transforms3d
def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True
    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

#from https://github.com/matthew-brett/transforms3d
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

removelatex="1"
def evaluate(eval_path, gt_path,blocpaperversion):
    print("File to be evaluated:{}".format(eval_path))
    axis = numpy.array([[0.0, 4.5, 3.2, 9.5], [-5, 10, -10, 20], [-1, 16, -7, 20]])
    setnum=0
    output_plot_file_name = "out.pdf"
    texfile = "latex.tex"
    if removelatex == "1":
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
            t=np.array([])
            Lines = gt_in.readlines()
            count=0
            for line in Lines:
            	a = line.split("   ")#line.split("   ")
            	if len(a ) == 1:
            		continue
            	digits2=a[0].split(".")
            	stringy=np.double(digits2[0]+'.'+digits2[1])
            	floata=np.array(a).astype(np.double)
            	floata[0]=stringy
            	if count == 0:
            		t=np.array(floata)
            		count=1
            	else:
            		t=np.vstack((t,np.array(floata)))
            return t

    def plot_data(data, str_color):
        plt.plot(data[:, 1], data[:, 2], str_color, linewidth=2)
        starting_point = (data[0, 1], data[0, 2])
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

    def m44(x,y,z):
        return numpy.array((1.0, 0.0, 0.0, x),(0.0, 1.0, 0.0, y),(0.0, 0.0, 1.0, z),(0.0, 0.0, 0.0, 1.0), dtype=numpy.float64)

    def exm44(data):
        return np.array((data[0,3],data[1,3],data[2,3]), dtype=numpy.float64)
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
    print("Read data had shape:",gt_data.shape,eval_data.shape)	
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
    first_legend=plt.legend(handles=patchList,fancybox=True,loc='upper left' ,framealpha=0.5, markerscale=1,handletextpad=0.,frameon=False,fontsize=24,bbox_to_anchor=(-0.05, 1.0))  # lower right .04, -0.04) upper left (-0.05, 1.04)
    skip = 1
    # http://akuederle.com/create-numpy-array-with-for-loop
    klast = 0
    gtlength=getpathlength(gt_data)
    evallength=getpathlength(eval_data)
    print(evallength)
    print(gtlength)
    error_array = np.array([])
    error_ang_array = np.array([])
    discon2array=np.array([])
    prevgt=[0,0,0]
    firsteval=1
    discon=0.0
    discon2=0.0
    disconcount=0
    done=0
    lastvalue=0
    if blocpaperversion == '1':
        lastvalue=-1
    #print(blocpaperversion,lastvalue)
    for i in range(0, np.size(gt_data, 0)):
        getmaxmin(gt_data[i],maxmin)#required for plotting
    for i in range(0, np.size(eval_data, 0)+lastvalue):
        getmaxmin(eval_data[i],maxmin) #required for plotting
        if eval_data[i, 0] < gt_data[0, 0]:
            print("no gt data available for this timestamp",eval_data[i, 0],gt_data[0, 0])
        else:
            if eval_data[i, 0] > gt_data[np.size(gt_data, 0) - 1, 0]:
                print("no more gt data available",eval_data[i, 0],gt_data[np.size(gt_data, 0) - 1, 0])
                break
            for k in range(klast, np.size(gt_data, 0) - 1):
                skip = 1
                if eval_data[i, 0] >= gt_data[k, 0] and eval_data[i, 0] < gt_data[k + 1, 0]:
                    #print( "match at",eval_data[i,0],gt_data[k,0],i,k,klast,np.size(gt_data, 0))
                    klast = k
                    skip = 0
                    break
                if skip == 1:
                    k = np.size(gt_data, 0)-1
                    if eval_data[i, 0] == gt_data[k, 0]:
                    	k=k-1
                    	skip = 0
                    	done =1 #avoids if the last two timestamps of the dataset are given twice those data gets evaluated
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
                   #first value cannot be a discontinuity
                   discon2array= np.append(discon2array,-1)
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
                   lengveceval2=math.sqrt(math.pow(veceval[0]-vecgt[0],2)+math.pow(veceval[1]-vecgt[1],2))
                   #Vecevaltf transforms eval data into the GT system the transform is calculated via ominus and applied on the eval data
                   #vecevaltf=exm44(np.matmul(ominus(m44(eval_data[i-1, 1],eval_data[i-1, 2]),m44(prevgt[0],prevgt[1])),m44(veceval[0],veceval[1])))
                   #lengveceval3=math.sqrt(math.pow(vecevaltf[0]-vecgt[0],2)+math.pow(vecevaltf[1]-vecgt[1],2))
                   if lenvecgt > 0.0001 :
                      discon=discon+math.pow(1-lengveceval/lenvecgt,2)
                      disco2value=math.pow(lengveceval2/lenvecgt,2)
                      discon2=discon2+disco2value
                      discon2array= np.append(discon2array,disco2value)
                      disconcount=disconcount+1
                   else:
                      discon2array= np.append(discon2array,-1)
                prevgt=[x_gt,y_gt,z_gt]
                #angle at k+1 and angle at current k
                angleskp1 = euler_from_quaternion([gt_data[k + 1, 4], gt_data[k + 1,5],gt_data[k + 1,6], gt_data[k + 1, 7]])
                anglesk = euler_from_quaternion([gt_data[k, 4], gt_data[k,5],gt_data[k,6], gt_data[k, 7]])
                #Changed Interpolation for angle calculation for ground truth at critical point when positive value for gt followed by a negative angle for gt, or vise versa. 
		#This requires bringing the negative angle in a positive value only system and then based on the outcome of the interpolation converting it back into a negative angle.
                if anglesk[2]>0 and angleskp1[2]<0 and (angleskp1[2]+math.pi*2-anglesk[2])<math.pi:
                	intermedian=angleskp1[2]+math.pi*2
                	anglez_gt=(intermedian)*a+(1-a)*anglesk[2] 
                	if anglez_gt > math.pi:
                	  anglez_gt=anglez_gt-2*math.pi
                elif anglesk[2]<0 and angleskp1[2]>0 and (anglesk[2]+math.pi*2-angleskp1[2])<math.pi:
                	intermedian=anglesk[2]+math.pi*2     
                	anglez_gt=(angleskp1[2])*a+(1-a)*intermedian 
                	if anglez_gt > math.pi:
                	  anglez_gt=anglez_gt-2*math.pi
                else:
                	anglez_gt=a*angleskp1[2]+(1-a)*anglesk[2]
                angleval= euler_from_quaternion([eval_data[i, 4], eval_data[i,5],eval_data[i,6], eval_data[i, 7]])
                eang=angleval[2]-anglez_gt
                if(eang>3.14159265359):
                	eang=2*3.14159265359-eang
                if(eang<-3.14159265359):
                	eang=-2*3.14159265359-eang
                error_ang_array = np.append(error_ang_array,abs(eang))
        if done == 1:
                break
    if len(error_array) == 0:
        print("No Data with matching timestamps")
        #plt.show()
        return
    rmse = np.sqrt(np.dot(error_array, error_array) / len(error_array))
    rmsear = np.sqrt(np.dot(error_ang_array, error_ang_array) / len(error_ang_array))
    #mean = np.mean(error_array)
    median = np.median(error_array)
    std = np.std(error_array)
    max = np.max(error_array)
    min = np.min(error_array)
    if(max < abs(min)):
    	print("Min > Max")
    	max=abs(min)
    #meanang = np.mean(error_ang_array)
    medianang = np.median(error_ang_array)
    stdang = np.std(error_ang_array)
    maxang=np.max(error_ang_array)
    minang=np.min(error_ang_array)
    if(maxang < abs(minang)):
    	maxang=abs(minang)
    rel_trans_error=[]
    rel_rot_error=[]
    rel_trans_errorperframe=[]
    rel_rot_errorperframe=[]
    lengt=[]
    rotgt=[]
    lenperframegt=[]
    rotperframegt=[]
    ccrt=0
    ccrs=0
    try:
    	print(gt_path,eval_path)
    	traj_gt = tr.read_trajectory(gt_path)
    	traj_est = tr.read_trajectory(eval_path)
    	result = tr.evaluate_trajectory(traj_gt, traj_est, 10000, True, 1.0, "s", 0.00, 1.00)
    	rel_trans_error = numpy.array(result)[:, 4]
    	rel_rot_error = numpy.array(result)[:, 5]
    	lengt = numpy.array(result)[:, 6]
    	rotgt = numpy.array(result)[:, 7]
    	rmset = numpy.sqrt(numpy.dot(rel_trans_error, rel_trans_error) / len(rel_trans_error))
    	rmser = numpy.sqrt(numpy.dot(rel_rot_error, rel_rot_error) / len(rel_rot_error))#we do not convert to degree!
    	ccl=np.divide(rel_trans_error,lengt)
    	x_ind=np.zeros(len(ccl))
    	for k in range(0,len(ccl)):
    		if rel_trans_error[k] < 0.001:
    			x_ind[k]=1
    		else:
    			x_ind[k]=ccl[k]<=0.5
    	ccrt=np.sum(x_ind)/x_ind.size*100
    	print("CCRT",np.sum(x_ind),x_ind.size)
    	#challenge cope score rotatioanl
    	ccr=np.divide(rel_rot_error,rotgt)
    	x_ind=np.zeros(len(ccr))
    	for k in range(0,len(ccr)):
    		if rel_rot_error[k] < 0.00873:
    			x_ind[k]=1
    		else:
    			x_ind[k]=ccr[k]<=0.5
    	ccrs=np.sum(x_ind)/x_ind.size*100
    	print("CCRS",np.sum(x_ind),x_ind.size)
    	print("fine")
    	#resultframewise = tr.evaluate_trajectory(traj_gt, traj_est, 10000, True, 1.0, "f", 0.00, 1.00)
    	#rel_rot_errorperframe = numpy.array(resultframewise)[:, 5]
	#rotperframegt = numpy.array(resultframewise)[:, 7]
    	#result2 = tr.evaluate_trajectory(traj_gt, traj_est, 10000, True, 1.0, "f", 0.00, 1.00)
    	#rel_trans_errorperframe = numpy.array(result2)[:, 4]
    	#rel_rot_errorperframe=numpy.array(result2)[:, 5]
    	#lenperframegt = numpy.array(result2)[:, 6]
    	#rotperframegt=numpy.array(result2)[:, 7]
    except:
    	rmser=0
    	rmset=0
    	print("Your data is not 1 second long = minimum length as relative error requires data of 1s timedifference")
    relpath = 1
    if gtlength != 0: 
    	relpath=evallength/gtlength
    discoty=0
    discoty2=0
    if disconcount != 0:
    	discoty=math.sqrt(discon)/disconcount
    	discoty2=math.sqrt(discon2/disconcount)
    with open(texfile, "a") as param:
        writing = tail[first+1:end] + " & {:.3f}".format(rmse) + " & {:.3f}".format(rmsear) + " & {:.3f}".format(rmset)+ " & {:.3f}".format(rmser) + " & {:.3f}".format(max)+ " & {:.3f}".format(maxang) + " & {:.3f}".format(median)+ " & {:.3f}".format(medianang) + " & {:.3f}".format(std)+ " & {:.3f}".format(stdang)+ " & {:.3f}".format(relpath*100.0)+ " & {:.3f}".format(discoty)+ " & {:.3f}".format(discoty2) +" & {:.3f}".format(ccrt)+" & {:.3f}".format(ccrs) + "\n" 
        param.write(writing)
    with open(texfile, "r+") as param:
        content = param.read()
        param.seek(0, 0)
        writing = "Algorithm & RMSE_t & RMSE_{\|r\|} & RMSE_{rt} & RMSE_{rr} & Max_t & Max_{\|r\|} & Median_t & Median_{\|r\|} & Std_t & Std_{\|r\|} & Pathratio & Discontinuity & D_2 & CC_{rt}^{1 mm} & CC_{rr}^{0.5^\circ}"    
        writing = writing +eval_path+ " \\\ \hline  \n"
        param.write(writing + content )
    print("Algorithm & RMSE_t & RMSE_{\|r\|} & RMSE_{rt} & RMSE_{rr} & Max_t & Max_{\|r\|} & Median_t & Median_{\|r\|} & Std_t & Std_{\|r\|} & Pathratio & Discontinuity & D_2 & CC_{rt}^{1 mm} & CC_{rr}^{0.5^\circ}")
    print(rmse,rmsear,rmset,rmser,max,maxang, median,medianang, std, stdang,relpath*100.0,discoty,discoty2,ccrt,ccrs)
    '''crs=mplcursors.cursor(hover=True)
    crs.connect("add", lambda sel: sel.annotation.set_text('Point {},{}'.format(sel.target[0], sel.target[1])))
    plt.show()'''
    #plt.xlim(maxmin[0]-1, maxmin[1]+1)
    #plt.ylim(maxmin[2]-1, maxmin[3]+1)
    '''Save Plot'''
    DefaultSize = f.get_size_inches()
    f.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
    f.savefig(output_plot_file_name, bbox_inches='tight')
    output_plot_file_name = tail[first+1:end]+".pdf"
    f.savefig(output_plot_file_name, bbox_inches='tight')
    print("Storing global errors")	
    error=np.column_stack([error_array,error_ang_array,discon2array])
    np.savetxt(output_plot_file_name[:-4]+".out",error, delimiter=' ',fmt='%.10f')
    #relativerot errors and trans errors values and corresponding lengt and rotgt in one file
    print("Storing rel errors")	
    relerror=np.column_stack([rel_trans_error, rel_rot_error, lengt,rotgt])
    np.savetxt(output_plot_file_name[:-4]+"rel.out", relerror, delimiter=' ',fmt='%.10f')
    print("Storing rel errors per frame")	
    #relerror=np.column_stack([rel_trans_errorperframe, rel_rot_errorperframe,lenperframegt,rotperframegt])
    #np.savetxt(output_plot_file_name[:-4]+"relframewise.out", relerror, delimiter=' ',fmt='%.10f')
if __name__ == "__main__":
    removelatex=sys.argv[3]
    blocsiepaper=sys.argv[4]
    if removelatex == "1":
        print(sys.argv[2],sys.argv[1],removelatex,blocsiepaper)
    evaluate(sys.argv[1], sys.argv[2],blocsiepaper)
