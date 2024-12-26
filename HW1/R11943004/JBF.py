
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        #print(img.shape)
        if len(guidance.shape) == 3:
            #print(guidance.shape)            #(316,316,3)
            channel=3
        else:
            #print(guidance.shape)            #(316,316)
            channel=1
        output = np.zeros(img.shape)

        # spatial kernel Gs
        i, j = np.mgrid[-self.pad_w:self.pad_w+1, -self.pad_w:self.pad_w+1]
        two_sigmas_square=(2 * self.sigma_s**2)
        two_sigmar_square=(2 * self.sigma_r**2)
        spatial_kernel = np.exp(np.divide(-(np.multiply(i,i) + np.multiply(j,j)),two_sigmas_square)  )     
        #Normalize 
        padded_guidance = padded_guidance.astype('float64')
        padded_guidance = np.divide(padded_guidance,255)
        padded_img = padded_img.astype('float64')

        for i in range(self.pad_w, padded_img.shape[0] - self.pad_w):
            for j in range(self.pad_w, padded_img.shape[1] - self.pad_w):
                patch = padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]
                patch_guidance = padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]
                Range_kernel = np.divide(np.multiply(-(patch_guidance - padded_guidance[i, j]),(patch_guidance - padded_guidance[i, j])), two_sigmar_square)

                if channel==3:
                    Range_kernel = Range_kernel.sum(axis=2)  #往channel方向加
                    Range_kernel = np.exp(Range_kernel)
                elif channel==1:  
                    Range_kernel = np.exp(Range_kernel)
                
                weights = np.multiply(spatial_kernel,Range_kernel)
                weights= np.divide(weights, weights.sum())
                #for c in range(3):拆迴圈來加速
                output[i-self.pad_w, j-self.pad_w, 0] = np.multiply(weights,patch[:,:,0]).sum()          
                output[i-self.pad_w, j-self.pad_w, 1] = np.multiply(weights,patch[:,:,1]).sum()          
                output[i-self.pad_w, j-self.pad_w, 2] = np.multiply(weights,patch[:,:,2]).sum()          

# =============================================================================不知道為甚麼比較慢= =
  #              weights_expanded = weights[:, :, np.newaxis]  
              
#                output_patch = np.sum(  np.multiply(weights_expanded , patch), axis=(0, 1)) 
 #               output[i-self.pad_w, j-self.pad_w, :] = output_patch 
# 
#                     
# =============================================================================

        output.astype(np.uint8)
                
        return np.clip(output, 0, 255).astype(np.uint8)
    

