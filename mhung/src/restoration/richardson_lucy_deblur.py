import numpy as np
from scipy.signal import convolve2d
from .motion_kernel import make_motion_kernel
from .base_filter import BaseRestorationFilter
import cv2

class RichardsonLucyMotionDeblurFilter(BaseRestorationFilter):
    def __init__(self, config):
        self.kernel_length = config.get('kernel_length', 15)
        self.kernel_angle = config.get('kernel_angle', 0.0)
        self.num_iter = config.get('num_iter', 20)
        self.clip_output = config.get('clip_output', True)
        self.denoise_after = config.get('denoise_after', False)

    def restore(self, image: np.ndarray) -> np.ndarray:
        is_uint8 = image.dtype == np.uint8
        img_f = image.astype(np.float32)
        if is_uint8:
            img_f /= 255.0

        kernel = make_motion_kernel(self.kernel_length, self.kernel_angle)
        
        restored = np.zeros_like(img_f)
        for c in range(img_f.shape[2]):
            restored[:, :, c] = self._richardson_lucy_channel(img_f[:, :, c], kernel, self.num_iter)

        if self.clip_output:
            restored = np.clip(restored, 0.0, 1.0)
            
        if is_uint8:
            restored = (restored * 255.0).astype(np.uint8)
            
        if self.denoise_after:
            # simple fastnlmeans
            if is_uint8:
                # opencv expects BGR for some fastNlMeans, but RGB is fine
                restored = cv2.fastNlMeansDenoisingColored(restored, None, 10, 10, 7, 21)
            else:
                restored = cv2.fastNlMeansDenoisingColored((restored*255).astype(np.uint8), None, 10, 10, 7, 21)
                restored = restored.astype(np.float32) / 255.0
                
        return restored

    def _richardson_lucy_channel(self, channel, kernel, num_iter, eps=1e-8):
        estimate = np.full_like(channel, 0.5)
        kernel_flip = np.flipud(np.fliplr(kernel))

        for _ in range(num_iter):
            # scipy.signal.convolve2d(..., boundary="symm") is equivalent to cv2.filter2D with BORDER_REFLECT
            # cv2.filter2D computes correlation, so we flip the kernel for convolution
            conv_est = cv2.filter2D(estimate, -1, kernel_flip, borderType=cv2.BORDER_REFLECT)
            relative_blur = channel / (conv_est + eps)
            
            # The back projection is convolution with the flipped kernel. 
            # Flipping the flipped kernel gives the original kernel for cv2 correlation.
            estimate *= cv2.filter2D(relative_blur, -1, kernel, borderType=cv2.BORDER_REFLECT)
            estimate = np.clip(estimate, 0.0, 1.0)

        return estimate
