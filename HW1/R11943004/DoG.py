import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_gaussian_images_per_octave = self.num_DoG_images_per_octave + 1
        self.dog_image_8=[]
    def get_getdogimage(self):
        return self.dog_image_8
    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        keypoints = []
        image_blurred=[]
        for i in range(10):
            image_blurred.append(image)

        image_blurred[0]=image
        image_blurred[1] = cv2.GaussianBlur(image_blurred[0], (0, 0), sigmaX=self.sigma)
        image_blurred[2] = cv2.GaussianBlur(image_blurred[0], (0, 0), sigmaX=self.sigma**2)
        image_blurred[3] = cv2.GaussianBlur(image_blurred[0], (0, 0), sigmaX=self.sigma**3)
        image_blurred[4] = cv2.GaussianBlur(image_blurred[0], (0, 0), sigmaX=self.sigma**4)
        image_blurred[5] = cv2.resize(image_blurred[4], (image_blurred[4].shape[1] // 2, image_blurred[4].shape[0] // 2), interpolation=cv2.INTER_NEAREST)

        image_blurred[6] = cv2.GaussianBlur(image_blurred[5], (0, 0), sigmaX=self.sigma)
        image_blurred[7] = cv2.GaussianBlur(image_blurred[5], (0, 0), sigmaX=self.sigma**2)
        image_blurred[8] = cv2.GaussianBlur(image_blurred[5], (0, 0), sigmaX=self.sigma**3)
        image_blurred[9] = cv2.GaussianBlur(image_blurred[5], (0, 0), sigmaX=self.sigma**4)
        gaussian_images=image_blurred.copy()
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(1, self.num_gaussian_images_per_octave):
            dog_image = cv2.subtract(gaussian_images[i], gaussian_images[i-1])
            dog_images.append(dog_image)
        for i in range(6, 5+self.num_gaussian_images_per_octave):
            dog_image = cv2.subtract(gaussian_images[i], gaussian_images[i-1])
            dog_images.append(dog_image)
# =============================================================================
#         print(len(gaussian_images))
#         print(len(dog_images))
# =============================================================================
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        # Find local extrema in DoG images
        for o in range(self.num_octaves):
            for i in range(1, self.num_DoG_images_per_octave - 1):
                if o==0:
                    prev_img, current_img, next_img = dog_images[i-1], dog_images[i], dog_images[i+1]
                if o==1:
                    prev_img, current_img, next_img = dog_images[i+3], dog_images[i+4], dog_images[i+5]
                # Find local extrema in the 3x3x3 neighborhood
                for y in range(1, current_img.shape[0] - 1):
                    for x in range(1, current_img.shape[1] - 1):
                        if self.is_maximum(prev_img, current_img, next_img, x, y) or self.is_minimum(prev_img, current_img, next_img, x, y):
                            keypoints.append((o,y , x))

        # Convert keypoints to input image size
        keypoints = self.convert_keypoints_to_input_size(keypoints)
        keypoints = np.array(keypoints)
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique


        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        keypoints = np.unique(keypoints, axis=0)
# =============================================================================
#         print(len(keypoints))
#         print(keypoints)
# =============================================================================
        self.dog_image_8=dog_images
        return keypoints

    def is_maximum(self, prev_img, current_img, next_img, x, y):
        pixel_val = current_img[y, x]

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if (pixel_val <= prev_img[y + dy, x + dx] or
                    pixel_val <= current_img[y + dy, x + dx] or
                    pixel_val <= next_img[y + dy, x + dx]):
                    return False
        if pixel_val<= prev_img[y,x] or pixel_val<= next_img[y,x]:
            return False
        if abs(pixel_val) <= self.threshold:
             return False
        return True
    def is_minimum(self, prev_img, current_img, next_img, x, y):
        pixel_val = current_img[y, x]

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if (pixel_val >= prev_img[y + dy, x + dx] or
                    pixel_val >= current_img[y + dy, x + dx] or
                    pixel_val >= next_img[y + dy, x + dx]):
                    return False
        if  pixel_val>= prev_img[y,x] or pixel_val>= next_img[y,x]:
            return False
        if abs(pixel_val) <= self.threshold:
            return False
        return True
    def convert_keypoints_to_input_size(self, keypoints):
        # Convert the (octave, x, y) coordinates to the original image size
        converted_keypoints = []
        for octave, x, y in keypoints:
            scale = 2**octave
            converted_keypoints.append((x * scale, y * scale))
        return converted_keypoints

