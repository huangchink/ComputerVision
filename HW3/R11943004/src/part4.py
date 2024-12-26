import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    w_min=0
    w_max=0
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        w_min  += im1.shape[1]  
        w_max   = w_min+im2.shape[1]  
        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.knnMatch(des1, des2, k=2)      
        points1=[]
        points2=[]  
        for m,n in matches:
            if m.distance < 0.45 * n.distance:
                points1.append(kp1[m.queryIdx].pt)
                points2.append(kp2[m.trainIdx].pt)
        points1 = np.array(points1)
        points2 = np.array(points2)
        # TODO: 2. apply RANSAC to choose best H
        iteration = 7777
        threshold = 1.5
        inliers_max = 0
        H_max = np.eye(3)
        for i in range(iteration):
            inliers=0
            random_points1 = np.zeros((4,2))
            random_points2 = np.zeros((4,2)) 
            for j in range(4):
                random_choose = random.randint(0, len(points1)-1)
                random_points1[j] = points1[random_choose]
                random_points2[j] = points2[random_choose]
            H = solve_homography(random_points2, random_points1)
            
            onerow = np.ones((len(points1),1))
            U = np.concatenate( (np.array(points2), onerow), axis=1) #[x,y,1]
            V = np.concatenate( (np.array(points1), onerow), axis=1)       
   
            V_hat_T =np.dot(H,U.T)                                #[x',
                                                                  # y',
                                                                  # 1']
            V_hat_T =np.divide(V_hat_T,V_hat_T[-1,:])
            V_hat = np.transpose(V_hat_T)#[x,y,1]
            
            err  = np.linalg.norm((V_hat-V)[:,:-1], ord=2, axis=1,keepdims=False) #(N,)
            for i in range(len(err)):
                if err[i] < threshold:
                    inliers=inliers+1

            if inliers > inliers_max:
                inliers_max = inliers
                H_max = H
        # TODO: 3. chain the homographies
        #print('inlineNmax',inlineNmax)
        last_best_H = last_best_H.dot(H_max)
        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0,  im2.shape[0], w_min, w_max, direction='b')
    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)