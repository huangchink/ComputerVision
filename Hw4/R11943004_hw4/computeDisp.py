import numpy as np
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    Il_pad = np.pad(Il, ((1,1),(1,1),(0,0)), 'constant')
    Ir_pad = np.pad(Ir, ((1,1),(1,1),(0,0)), 'constant')
    Il_binmap = np.zeros((h,w,ch,8),dtype=bool)
    Ir_binmap = np.zeros((h,w,ch,8),dtype=bool) #h x w x c x d
    offsety= np.array([-1,-1,-1,
                       0 ,   0,
                       1 ,1 ,1])
    offsetx = np.array([-1,0,1,
                        -1,  1,
                        -1,0,1])
    for i in range(8):
        Il_binmap[:,:,:,i]=  Il[:,:,:] < Il_pad[1+offsety[i]:h+1+offsety[i],1+offsetx[i]:w+1+offsetx[i] , :]
        Ir_binmap[:,:,:,i]=  Ir[:,:,:] < Ir_pad[1+offsety[i]:h+1+offsety[i],1+offsetx[i]:w+1+offsetx[i] , :]
    census_cost_l2r = np.zeros((h,w,max_disp+1))
    census_cost_r2l = np.zeros((h,w,max_disp+1))
    for disp in range(max_disp+1):
        l_range = Il_binmap[:,disp:w,:,:]
        r_range = Ir_binmap[:,:w-disp,:,:]

        hamming_distance = np.sum(l_range^r_range,axis = 3)  # hamming distance of the 4th dimension (axis = 3)
        census_cost_l2r[:,disp:w,disp] = np.sum(hamming_distance,axis = 2)                                     # sum hamming distance of 3 channels
        census_cost_l2r[:,0:disp,disp] = np.repeat(census_cost_l2r[:,disp,disp].reshape(-1,1),disp,axis = 1) # repeat edge costs for out-of-bound pixels 
        census_cost_r2l[:,0:w-disp,disp] = np.sum(hamming_distance,axis = 2)
        census_cost_r2l[:,w-disp:w,disp] = np.repeat(census_cost_r2l[:,disp,disp].reshape(-1,1),disp,axis = 1)


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    # Joint bilateral filter
    for disp in range(max_disp+1):
        census_cost_l2r[:,:,disp] = xip.jointBilateralFilter(Il.astype(np.float32), census_cost_l2r[:,:,disp].astype(np.float32), 15, 40, 30)
        census_cost_r2l[:,:,disp] = xip.jointBilateralFilter(Ir.astype(np.float32), census_cost_r2l[:,:,disp].astype(np.float32), 15, 40, 30)

    # 
    # cost_l2r = xip.guidedFilter(guide=Il.astype(np.float32), src=cost_l2r.astype(np.float32), radius=2, eps=100, dDepth=-1)
    # cost_r2l = xip.guidedFilter(guide=Ir.astype(np.float32), src=cost_r2l.astype(np.float32), radius=2, eps=100, dDepth=-1)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    DL = np.argmin(census_cost_l2r,axis = 2).astype(np.uint8)
    Dr = np.argmin(census_cost_r2l,axis = 2).astype(np.uint8)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    # Left-right consistency check
    for y in range(h):
        for x in range(w):
            if (DL[y,x] != Dr[y,x-DL[y,x]]) and (x-DL[y,x])>=0  :
                DL[y,x] = 0 #make hole

    # Hole filling
    FL = np.copy(DL)
    FR = np.copy(DL)
    rows=np.where(FL[:,0]==0)
    FL[rows, 0] = max_disp # pad maximum for the holes in boundary (start point)
    rows=np.where(FR[:,w-1]==0)
    FR[rows,w-1] = max_disp
    for x in range(1,w):
        rows=np.where(FL[:,x]==0)
        FL[rows,x] = FL[rows, x-1]
    for x in range(w-2,-1,-1):
        rows=np.where(FR[:,x]==0)
        FR[rows,x] = FR[rows, x+1]
    labels = np.minimum(FL,FR)

    # Weighted median filtering
    # labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels.astype(np.uint8), 15, 5, xip.WMF_COS    )

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels.astype(np.uint8), 15, 5, xip.WMF_JAC   )

    return labels.astype(np.uint8)