from .laplacian_pyramid import *
import numpy as np
import math
import matplotlib.pyplot as plt

__all__ = [
    "imgFD", "encode", "decode", "gettbit", "rmsError", "plotImg", "RMSdiff",
    "findSteps", "compRatio", "impRes"
]


def imgFD(func, img: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """
    Function to filter/decimate row and column of an image
    
    Parameters:
        func: Function that filters (rowint) or decimate (rowdec)
        img: Image to be filtered/decimated
        filt: Filter coefficients
        
    Returns:
        img2: Filtered/Decimated image
        
    """
    # Filter/Decimate rows
    img1 = func(img, filt)
    # Filter/Decimate cols
    img2 = func(img1.T, filt).T
    return img2


def encode(img, filt, layers):
    """
    Encodes image X into laplacian pyramid by splitting into 
    #(layers) of high pass images and 1 low pass image
    
    Parameters:
        X: Image matrix
        h: Filter coefficients
        layers: Number of layers in Laplacian Pyramid
    Returns:
        Ys: List of high pass images, [Y0, Y1, ...]
        Xs: Remaining low pass image

    """
    Xs = [img]
    Ys = []
    for i in range(0, layers):
        X = imgFD(rowdec, Xs[i], filt)
        Xs.append(X)
        Z = imgFD(rowint, Xs[i + 1], 2 * filt)
        Y = Xs[i] - Z
        Ys.append(Y)
    return Ys, Xs[-1]


def decode(Ys, Xn, filt):
    """
    Decode compressed image with list of high pass images Ys and 1 low pass image Xn
    
    Parameters:
        Ys: List of high pass images, [Y0, Y1, ...]
        Xn: Remaining low pass image
        h: Filter coefficients
    Returns:
        Zs: Reconstructed images at each stage, [Z0, Z1, ...]
        
    """
    layers = len(Ys)
    Zsr = [Xn]
    for i in range(0, layers):
        Z = imgFD(rowint, Zsr[i], 2 * filt) + Ys[layers - 1 - i]
        Zsr.append(Z)
    Zs = Zsr[::-1]
    return Zs


def plotImg(
    imgs,
    cols,
    title='Image: ',
    index=[],
    scale=4,
    cmap=None,
    save=False,
    name='Image.png',
    dest='D:\\Cambridge\\Part IIA\\Projects\\SF2-Image-Processing\\Reports\\'
):

    rows = math.ceil(len(imgs) / cols)

    # Check if length of index == length of images
    if len(index) == 0:
        index = np.arange(0, len(imgs), 1)
    elif len(index) != len(imgs):
        raise Exception('Different lengths of indices and images!')
    else:
        pass

    # Plot images on subplots
    fig, ax = plt.subplots(rows, cols, figsize=(scale * cols, scale * rows))

    if rows == 1 and cols == 1:
        ax.set_title(title + str(index[0]))
        ax.imshow(imgs[0], cmap=cmap)
        ax.axis('off')

    elif rows == 1 or cols == 1:
        for i in range(0, len(imgs)):
            ax[i].set_title(title + str(index[i]))
            ax[i].imshow(imgs[i], cmap=cmap)
            ax[i].axis('off')
    else:
        for i in range(0, len(imgs)):
            row = int(i / cols)
            col = i % cols
            ax[row][col].set_title(title + str(index[i]))
            ax[row][col].imshow(imgs[i], cmap=cmap)
            ax[row][col].axis('off')
    
    fig.subplots_adjust(wspace=0, hspace=0, top = 1)
    # Save image to destination
    if save:
        plt.savefig(dest + '\\' + name, bbox_inches='tight', pad_inches = 0.0)

    # Display image on console
    plt.show()


def gettbit(img, filt, level, step):
    """
    Get total bits for specified laplacian pyramid level and quantisation step size
    
    Parameters:
        img: Image to be compressed
        filt: Filter coefficients
        level: Number of layers in Laplacian pyramid
        step: List of quantisation step size
        
    Returns:
        tbit: Total number of bits to encode
        
    """
    assert len(
        step
    ) == level + 1, "Size of step array not compatible with number of layers"
    tbit = 0
    Ys, Xn = encode(img, filt, level)
    Ys.append(Xn)
    for i in range(0, len(Ys)):
        Y = Ys[i]
        pixels = Y.shape[0] * Y.shape[1]
        bitspp = bpp(quantise(Y, step[i]))
        bits = bitspp * pixels
        tbit += bits
    return tbit


def rmsError(img, filt, layer, step, indiv=False):
    """
    Returns RMS error for quantised laplacian pyramid of specified level (rms), and decoded image (Zs[0])
    
    Parameters:
        img: Image matrix
        filt: Filter coefficients
        layers: Number of layers in Laplacian Pyramid
        step: List of quantisation step sizes
        indiv: False: Return total RMS, True: Return individual RMS of layers
        
    Returns:
        rms: Root Mean Squared error of quantized image at desired layer and step size
             Type: Integer if indiv ==  False, Array if indiv == True
        Zs[0]: Decoded image (to original size)
        
    """
    assert len(
        step
    ) == layer + 1, "Size of step array not compatible with number of layers"

    # Calculate total RMS value across all layers
    if not indiv:
        Ys, Xn = encode(img, filt, layer)
        Xq = quantise(Xn, step[-1])
        Qs = []  # Quantised images of Ys
        for j in range(0, layer):
            Qs.append(quantise(Ys[j], step[j]))
        Zs = decode(Qs, Xq, filt)
        rms = np.std(img - Zs[0])

    # Calculate RMS contribution by individual layers
    # Decoded image not relevant
    else:
        rms = np.zeros(layer + 1)
        # Quantize individual Yn layer and calculate RMS
        for i in range(0, layer):
            Ys, Xn = encode(img, filt, layer)
            Ys[i] = quantise(Ys[i], step[i])
            Zs = decode(Ys, Xn, filt)
            rms[i] = np.std(img - Zs[0])

        # Quantize Xn only and calculate RMS
        Ys, Xn = encode(img, filt, layer)
        Xq = quantise(Xn, step[-1])
        Zs = decode(Ys, Xq, filt)
        rms[-1] = np.std(img - Zs[0])

    return rms, Zs[0]


def RMSdiff(img, filt, level, step, rmsRef):
    """
    Returns absolute difference in RMS compared to given reference value
    
    Parameters:
        step: Quantisation step size
        img: Image Matrix
        filt: Filter coefficients
        level: Laplacian Pyramid level
        rmsRef: Reference RMS value
        
    Returns:
        rmsdiff: Absolute difference in RMS compared to rmsRef
        
    """
    rms, D = rmsError(img, filt, level, step)
    rmsdiff = abs(rms - rmsRef)
    return rmsdiff


def findSteps(img, filt, levels, stepratio, initGuess, rmsRef, disp = False):
    """
    Function to optimise quantisation step size w.r.t. rmsRef
    
    Parameters:
        img: Image Matrix
        filt: Filter coefficients
        levels: List of Laplacian pyramid levels to find optimal step sizes for
        stepratio: List of lists of stepratios for different levels
        initGuess: List of initial guesses scale of quantisation step sizes for respective levels.
        rmsRef: Reference RMS value
        
    Returns:
        steps: Dictionary of levels, optimal quantisation step sizes, RMS values, and RMS diff for given list of levels
        Ds: List of decoded quantized images at respective optimal step sizes
        
    """
    steps = {}
    Ds = []
    tol = 5e-3

    for i in range(0, len(levels)):
        # Initialise
        level = levels[i]
        guess = initGuess[i]
        step = np.array(stepratio[i])
        err = RMSdiff(img, filt, level, step * guess, rmsRef)

        while err > tol and guess > 0:
            err = RMSdiff(img, filt, level, step * guess, rmsRef)
            guess -= 5e-4
            if disp:
                print(guess, err)

        rms, D = rmsError(img, filt, level, step * guess)
        steps[level] = {
            'Ratio': guess,
            'Step': step * guess,
            'RMS': rms,
            'RMS Diff': err
        }
        Ds.append(D)
    return steps, Ds


def compRatio(img, filt, levels, stepsizes):
    """
    Returns compression ratio of image compressed at specified levels and step sizes
    
    Parameters:
        img: Image Matrix
        filt: Filter coefficients
        levels: Desired pyramid levels to find compression ratio. Include reference level in first element
        stepsizes: Desired step sizes at corresponding pyramid levels. Include reference level in first element
        
    Returns:
        ratio: Lists of compression ratios at desired levels and step sizes
        
    """
    ratio = np.empty(len(levels))
    refBit = gettbit(img, filt, levels[0], stepsizes[0])
    for i in range(0, len(levels)):
        compBit = gettbit(img, filt, levels[i], stepsizes[i])
        ratio[i] = refBit / compBit
    assert (ratio[0] == 1)
    return ratio


def impRes(img, filt, layer):
    """
    Get impulse response of each layer
    
    Parameters:
        img: Image matrix
        layer: number of layers in laplacian pyramid (No. of high pass images)
        
    Returns:
        Ze: List of energy in the overall image as a result of impulse in each layer
            Length: layer+1 (for the impulse of Xn)
        
    """
    layer += 1
    Ze = np.zeros(layer)

    for i in range(0, layer):
        Ys, Xn = encode(img, filt, layer)
        rows, cols = Ys[i].shape
        rmid = int(rows / 2)
        cmid = int(cols / 2)
        Ys[i][rmid][cmid] = 100

        assert (np.amax(Ys[i]) == 100)

        Zs = decode(Ys, Xn, filt)
        Ze[i] = np.sum(Zs[0]**2.0)

    return Ze
