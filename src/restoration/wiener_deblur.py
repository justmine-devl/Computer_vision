import numpy as np
from scipy.fft import fft2, ifft2
from .motion_kernel import make_motion_kernel

class WienerMotionDeblurFilter:
    def __init__(self, config):
        self.kernel_length = config.get('kernel_length', 15)
        self.kernel_angle = config.get('kernel_angle', 0.0)
        self.lambd = config.get('regularization_lambda', 0.01)
        self.clip_output = config.get('clip_output', True)
        self.post_sharpen = config.get('post_sharpen', False)
        self.post_sharpen_amount = config.get('post_sharpen_amount', 0.0)

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Convert to float32 [0, 1]
        is_uint8 = image.dtype == np.uint8
        img_f = image.astype(np.float32)
        if is_uint8:
            img_f /= 255.0

        kernel = make_motion_kernel(self.kernel_length, self.kernel_angle)
        
        restored = np.zeros_like(img_f)
        for c in range(img_f.shape[2]):
            restored[:, :, c] = self._wiener_deconv_channel(img_f[:, :, c], kernel, self.lambd)

        if self.post_sharpen and self.post_sharpen_amount > 0:
            import cv2
            blurred = cv2.GaussianBlur(restored, (0, 0), 3.0)
            restored = restored + self.post_sharpen_amount * (restored - blurred)

        if self.clip_output:
            restored = np.clip(restored, 0.0, 1.0)
            
        if is_uint8:
            restored = (restored * 255.0).astype(np.uint8)
            
        return restored

    def _wiener_deconv_channel(self, channel, kernel, balance):
        H = fft2(kernel, s=channel.shape)
        B = fft2(channel)
        H_conj = np.conj(H)
        restored = np.real(ifft2((H_conj / (np.abs(H)**2 + balance)) * B))
        return restored
