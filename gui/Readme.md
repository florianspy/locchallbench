1.  Download the files evaluate_rpe.py and evaluate_ate.py from https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/
2.  Add in evaluate_ate.py the following function
```#no aligning
def align2(model,data):
    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)
    alignment_error = model - data
    #hier werden dann für die einzelnen punkte die translation fehler berechnet also sqrt(x^2 +y^2) so wie wir es auch machen
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error```
3.  And change line
    ```rot,trans,trans_error = align(second_xyz,first_xyz)```
  to
    ```rot,trans,trans_error = align2(second_xyz,first_xyz)```
4.  Put the files into the same folder as main.py
5. To run write python3 main.py
