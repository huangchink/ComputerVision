import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []
    #u[i,0]=ux for n
    #u[0,i]=ux for n
    for n in range(N):
        A.append([u[n, 0], u[n, 1], 1,  0,  0,  0, -u[n, 0]*v[n, 0], -u[n, 1]*v[n, 0], -v[n, 0]])
        A.append([0, 0, 0, u[n, 0], u[n, 1], 1,  -u[n, 0]*v[n, 1],  -u[n, 1]*v[n, 1],  -v[n, 1]])
    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A)
    
    # check
    #print(U)
    #print(S)
    #print(VT)
    #last row of VT = last column of V    divide by z to be in homogenous coordinate
    h = VT[-1,:]/VT[-1,-1]
    H = h.reshape(3, 3)
    return H

def interpolate(src_img, x, y):
    # Floor to get the integer parts and ensure types are correct
    x_base = np.floor(x).astype(int)
    y_base = np.floor(y).astype(int)

    # Clip to ensure coordinates are within the image bounds
    x_base = np.clip(x_base, 0, src_img.shape[1] - 2)
    y_base = np.clip(y_base, 0, src_img.shape[0] - 2)
    x_top = x_base + 1
    y_right = y_base + 1

    # Fractional parts for interpolation
    x_fraction = x - x_base
    y_fraction = y - y_base

    # Reshape for broadcasting
    x_fraction = x_fraction[:, np.newaxis]
    y_fraction = y_fraction[:, np.newaxis]

    # Interpolate
    top_left = src_img[y_base, x_base]
    top_right = src_img[y_base, x_top]
    bottom_left = src_img[y_right, x_base]
    bottom_right = src_img[y_right, x_top]

    top = (1 - x_fraction) * top_left + x_fraction * top_right
    bottom = (1 - x_fraction) * bottom_left + x_fraction * bottom_right
    interpolated_values = (1 - y_fraction) * top + y_fraction * bottom

    return interpolated_values


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    x, y = x.reshape(-1,1), y.reshape(-1,1)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    canvas   = np.concatenate((x, y, np.ones((x.shape))), axis = 1)    #[x,y,1]
    canvas_T = canvas.T #[x,    ]
                        # y,
                        # 1,
    #print(canvas.shape)
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        back_homogeneous = H_inv.dot(canvas_T)#[x',    ]
                                              # y',
                                              # z',
        src_back_points =  np.divide(back_homogeneous , back_homogeneous[-1,:])[:-1].T
        #bilinear
        # Get the interpolated pixel values
        pixel_values = interpolate(src, src_back_points[:, 0], src_back_points[:, 1])





        # # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        # Use boolean indexing to filter out-of-bound points
        mask = np.logical_and(np.logical_and(src_back_points[:,0] >= 0, src_back_points[:,0] < w_src), 
                          np.logical_and(src_back_points[:,1] >= 0, src_back_points[:,1] < h_src))
        valid_dst_points = canvas[mask].astype(int)  # Ensure points are within src range

        # # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # valid_src_back_points = src_back_points[mask].astype(int)
        pixel_values_in_bound = pixel_values[mask]



        # # TODO: 6. assign to destination image with proper masking
        # Assign pixel values to the destination image using array indexing
        dst[valid_dst_points[:, 1], valid_dst_points[:, 0]] = pixel_values_in_bound
        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        forward_homogeneous = H.dot(canvas_T)#[x',    ]
                                             # y',
                                             # z',
        forward_points = np.divide(forward_homogeneous , forward_homogeneous[-1,:])[:-1].T
        #nearest neighbor
        forward_points = np.round(forward_points).astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.logical_and(np.logical_and(forward_points[:,0] >= 0, forward_points[:,0] < w_dst), 
                          np.logical_and(forward_points[:,1] >= 0, forward_points[:,1] < h_dst))
        # TODO: 5.filter the valid coordinates using previous obtained mask
        valid_src_points = canvas[mask].astype(int)  # Ensure points are within src range
        valid_dst_points = forward_points[mask]
        
        # TODO: 6. assign to destination image using advanced array indicing
        dst[valid_dst_points[:, 1], valid_dst_points[:, 0]] = src[valid_src_points[:, 1], valid_src_points[:, 0]]

        pass

    return dst 
