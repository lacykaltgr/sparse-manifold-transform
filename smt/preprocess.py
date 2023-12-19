import cv2
import numpy as np

def downsample(data, algorithm="gaussian_pyramid", factor=2):

    if algorithm == 'steerable_pyramid':
        # steerable pyramid downsample
        # https://pyrtools.readthedocs.io/en/latest/tutorials/03_steerable_pyramids.html
        #filt = 'sp3_filters' # There are 4 orientations for this filter
        #pyr = pt.pyramids.SteerablePyramidSpace(video[0], height=4, order=3)
        raise NotImplementedError()

    elif algorithm == 'gaussian_pyramid':
        downsampled = []
        for frame in data:
            for i in range(factor):
                frame = cv2.pyrDown(frame)
            downsampled.append(frame)
        downsampled = np.array(downsampled)
        print("Original shape: ", data.shape,
              "Downsampled shape: ", downsampled.shape)

    else:
        raise ValueError("Unknown algorithm: ", algorithm)


def normalize(data, method='rgb'):
    if method == 'minmax':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'mean':
        data = (data - np.mean(data)) / np.std(data)
    elif method == 'rgb':
        data = (data.astype(np.float32) - 127.5) / 127.5
    else:
        raise ValueError("Unknown normalization method: ", method)
    return data


def whiten_frame(frame):
    # Take Fourier transform
    f_transform = np.fft.fft2(frame)

    # Create frequency grid
    rows, cols = frame.shape
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)

    # Create meshgrid from frequency grid
    u, v = np.meshgrid(u, v)

    # Calculate frequency radius
    r = np.sqrt(u**2 + v**2)

    # Define linear ramp function w1(u, v) = r
    w1 = r

    # Define low-pass windowing function w2(u, v) = e^(-(r/r0)^4)
    r0 = 48
    w2 = np.exp(-(r/r0)**4)

    # Calculate whitening mask function w(u, v) = w1(u, v) * w2(u, v)
    whitening_mask = w1 * w2

    # Modulate amplitude by whitening mask
    f_transform_whitened = f_transform * whitening_mask

    # Take inverse Fourier transform
    frame_whitened = np.fft.ifft2(f_transform_whitened).real

    return frame_whitened


def whitening(data, method='fft'):
    whitened = []
    for frame in data:
        whitened.append(whiten_frame(frame))
    whitened = np.array(whitened)
    return whitened