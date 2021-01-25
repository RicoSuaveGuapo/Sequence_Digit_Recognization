import math
import time
import os

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def band_rejecter(singal_fla, sampling_rate=101, width_ratio=0.2):
    '''
    ref: https://tomroelandts.com/articles/how-to-create-simple-band-pass-and-band-reject-filters
    Note:
        sampling_rate is better to be odd
    '''
    # sampling
    hist, bin_edges = np.histogram(singal_fla, bins=sampling_rate)
    bin_edges       = bin_edges[:-1].astype(np.int)
    hist            = hist.astype(np.int)
    singal          = np.stack((bin_edges, hist), axis=1)

    median_index  = [i for i, d in enumerate(hist) if d == np.max(hist)][0]
    index_width   = int(median_index*width_ratio)
    
    fL   = (median_index-index_width)/float(sampling_rate)
    fH   = (median_index+index_width)/float(sampling_rate)
    mode = np.arange(sampling_rate)

    # low-pass filter
    hlp  = np.sinc(2 * fL * (mode - (sampling_rate - 1) / 2))
    hlp *= np.blackman(sampling_rate)
    hlp /= np.sum(hlp)

    # high-pass filter
    hhp = np.sinc(2 * fH * (mode - (sampling_rate - 1) / 2))
    hhp *= np.blackman(sampling_rate)
    hhp /= np.sum(hhp)
    hhp = -hhp
    hhp[(sampling_rate - 1) // 2] += 1

    h = hlp + hhp

    s = np.convolve(hist, h)
    # print(bin_edges[median_index], bin_edges[median_index+index_width], bin_edges[median_index-index_width])
    print(s)
    
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[max(y), mean, sigma])
    return popt

def hough(img):
    h, w     = img.shape
    v_0, u_0 = int(h/2), int(w/2)

    # compute the theta
    theta = []
    for u in range(w):
        for v in range(h):
            if img[v, u] == 255:
                try:
                    ratio = (v-v_0)/(u-u_0)
                except:
                    continue
                if ratio > 0:
                    theta.append(np.arctan(ratio))
                elif ratio <0:
                    theta.append(np.pi + np.arctan(ratio))
            else:
                continue

    return theta

def idft(img_dft_filtered):
    img_dft_ishifted = np.fft.ifftshift(img_dft_filtered)
    img_idft = cv2.idft(img_dft_ishifted)
    img_idft = cv2.magnitude(img_idft[:,:,0],img_idft[:,:,1])
    return img_idft

def maxThetaCounter(theta_list, rows, cols, radius=130, list_of_list=False):
    if not list_of_list:
        hist, bin_edges = np.histogram(theta_list, bins=180, density=True)
        bin_centres     = (bin_edges[:-1] + bin_edges[1:])/2
        max_angle       = [bin_centres[idx] for idx, count in enumerate(hist) if count == max(hist)][0]
        line_upper_pt   = (int(rows/2 - radius*np.cos(max_angle)), int(cols/2 - radius*np.sin(max_angle)))
        line_lower_pt   = (int(rows/2 + radius*np.cos(max_angle)), int(cols/2 + radius*np.sin(max_angle)))
        return line_upper_pt, line_lower_pt

    else:
        line_upper_pts = []
        line_lower_pts = []
        for t_list in theta_list:
            hist, bin_edges = np.histogram(t_list, bins=180, density=True)
            bin_centres     = (bin_edges[:-1] + bin_edges[1:])/2
            centre_hist     = [hist[idx] for idx, d in enumerate(bin_centres) if d > 1 and d < 2]
            bin_centres     = [d for d in bin_centres if d > 1 and d < 2]
            max_angle       = [bin_centres[idx] for idx, count in enumerate(centre_hist) if count == max(centre_hist)][0]
            line_upper_pts.append((int(rows/2 - radius*np.cos(max_angle)), int(cols/2 - radius*np.sin(max_angle))))
            line_lower_pts.append((int(rows/2 + radius*np.cos(max_angle)), int(cols/2 + radius*np.sin(max_angle))))

        return line_upper_pts, line_lower_pts

def dft_denoiser(path, des_path=None):
    if os.path.isfile(path):
        image_path = path
        img_src    = cv2.imread(image_path,0)
        rows, cols = img_src.shape

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph  = cv2.morphologyEx(img_src, cv2.MORPH_CLOSE, kernel)
        morph  = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        img_dft         = cv2.dft(np.float32(morph), flags = cv2.DFT_COMPLEX_OUTPUT)
        # output 2 channels
        # 1sr channel is the real part
        # 2nd channel is the im   part
        img_dft_shifted = np.fft.fftshift(img_dft)
        img_dft_log     = 20*np.log(cv2.magnitude(img_dft_shifted[:,:,0],img_dft_shifted[:,:,1]))

        # ---------- paper method ----------
        img_dft_log     = img_dft_log.astype(np.uint8)
        ret2, ostu = cv.threshold(img_dft_log,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        theta_list = hough(ostu)
        theta_list = np.array(theta_list)
        line_upper_pt, line_lower_pt = maxThetaCounter(theta_list, rows=rows, cols=cols, radius=130, list_of_list=False)
        
        # apply mask
        img_dft_filtered     = cv2.line( img_dft_shifted, line_upper_pt, line_lower_pt, color=0, thickness=5)
        img_dft_filtered_log = 20*np.log(cv2.magnitude(img_dft_filtered[:,:,0],img_dft_filtered[:,:,1]))

        # IDFT back
        img_idft = idft(img_dft_filtered)
        plt_name = ['Original', 'Morph open/close', 'DFT', 'Binary w/ OSTU', 'Cut off max A(theta)', 'Output']
        f, axs = plt.subplots(2,3)
        f.set_figheight(10)
        f.set_figwidth(70)
        axs[0,0].imshow(img_src)
        axs[0,0].set_title(plt_name[0])
        axs[0,1].imshow(morph)
        axs[0,1].set_title(plt_name[1])
        axs[0,2].imshow(img_dft_log)
        axs[0,2].set_title(plt_name[2])
        axs[1,0].imshow(ostu)
        axs[1,0].set_title(plt_name[3])
        axs[1,1].imshow(img_dft_filtered_log)
        axs[1,1].set_title(plt_name[4])
        axs[1,2].imshow(img_idft)
        axs[1,2].set_title(plt_name[5])
        plt.show()
        
    elif os.path.isdir(path):
        if not os.path.exists(des_path):
            os.mkdir(des_path)

        since = time.time()
        image_paths = os.listdir(path)
        image_paths = [os.path.join(path, d) for d in image_paths if d.endswith('.bmp')]
        img_srcs    = [cv2.imread(d,0) for d in image_paths]
        rows, cols = img_srcs[0].shape

        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphs  = [cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel) for d in img_srcs]
        morphs  = [cv2.morphologyEx(d, cv2.MORPH_OPEN, kernel) for d in morphs]

        img_dfts         = [cv2.dft(np.float32(d), flags = cv2.DFT_COMPLEX_OUTPUT) for d in morphs]
        # output 2 channels
        # 1sr channel is the real part
        # 2nd channel is the im   part
        img_dft_shifteds = [np.fft.fftshift(d) for d in img_dfts]
        img_dft_logs     = [20*np.log(cv2.magnitude(d[:,:,0],d[:,:,1])) for d in img_dft_shifteds]

        # ---------- paper method ----------
        img_dft_logs = [d.astype(np.uint8) for d in img_dft_logs]
        ostus        = [cv.threshold(d,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1] for d in img_dft_logs]

        theta_lists = [hough(d) for d in ostus]
        theta_lists = [np.array(d) for d in theta_lists]

        line_upper_pts, line_lower_pts = maxThetaCounter(theta_lists, rows=rows, cols=cols, radius=130, list_of_list=True)
        # apply mask
        img_dft_filtereds = [cv2.line(img_dft_shifteds[idx], d, line_lower_pts[idx], color=0, thickness=5) for idx, d in enumerate(line_upper_pts)]
        # img_dft_filtered_logs = [20*np.log(cv2.magnitude(d[:,:,0],d[:,:,1])) for d in img_dft_filtereds] 


        # IDFT back
        img_idfts = [idft(d) for d in img_dft_filtereds]
        img_idfts = [(255*d/(np.max(d)-np.min(d))).astype(np.uint8) for d in img_idfts]
        [cv2.imwrite(os.path.join(des_path, os.path.basename(image_paths[idk])), d) for idk, d in enumerate(img_idfts)]
        print(f'One image spends {(time.time() - since)/len(img_idfts):.3f} sec')
        print(f'Total spend {time.time() - since:.3f} sec')



if __name__ == '__main__':
    image_path = 'data/20201229/EXT/resize/20201229080615_0EXT.bmp'
    img_src = cv2.imread(image_path,0)
    rows, cols = img_src.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph  = cv2.morphologyEx(img_src, cv2.MORPH_CLOSE, kernel)
    morph  = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    img_dft         = cv2.dft(np.float32(morph), flags = cv2.DFT_COMPLEX_OUTPUT)
    img_dft_shifted = np.fft.fftshift(img_dft)
    img_dft_log     = 20*np.log(cv2.magnitude(img_dft_shifted[:,:,0],img_dft_shifted[:,:,1]))

    # ----- for band reject -----
    # img_dft_log_fla = img_dft_log.flatten()
    # band_rejecter(img_dft_log_fla)
    # plt.hist(img_dft_log_ban, bins='auto')
    # plt.hist(img_dft_log_fla, bins=100)
    # plt.show()

    # ---------- high frequency cutoff ----------
    img_dft_log_fla = img_dft_log.flatten()
    hist, bin_edges = np.histogram(img_dft_log_fla, bins=100)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    A, x0, sigma = gauss_fit(bin_centres, hist)
    coeff = [A, x0, sigma]
    hist_fit = gauss(bin_centres, *coeff)
    plt.plot(bin_centres, hist_fit,  label='Fitted data')
    plt.plot(bin_centres, hist, '.',label='Test data')
    plt.legend()
    plt.show()
    # threshold = x0 + 5*sigma
    # img_dft_filtered_log = np.where(img_dft_log < threshold, img_dft_log, 0)
    # plt.imshow(img_dft_filtered_log)
    # plt.show()
    # mask    = np.ones((rows, cols,2), np.uint8)
    # mask[...,0] = np.where(img_dft_log < threshold, 1, 0)
    # mask[...,1] = mask[...,0]
    # img_dft_filtered    = img_dft_shifted*mask

    # ---------- paper method ----------
    # img_dft_log     = img_dft_log.astype(np.uint8)
    # ret2, ostu = cv.threshold(img_dft_log,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # theta_list = hough(ostu)
    # theta_list = np.array(theta_list)
    # hist, bin_edges = np.histogram(theta_list, bins=180, density=True)
    # max_angle = [bin_edges[idx] for idx, count in enumerate(hist) if count == max(hist)][0]
    # radius          = 130
    # line_upper_pt   = (int(rows/2 - radius*np.cos(max_angle)), int(cols/2 - radius*np.sin(max_angle)))
    # line_lower_pt   = (int(rows/2 + radius*np.cos(max_angle)), int(cols/2 + radius*np.sin(max_angle)))
    # apply mask
    # img_dft_filtered     = cv2.line( img_dft_shifted, line_upper_pt, line_lower_pt, color=0, thickness=5)
    # img_dft_filtered_log = 20*np.log(cv2.magnitude(img_dft_filtered[:,:,0],img_dft_filtered[:,:,1]))


    # --------- IDFT back ---------
    # img_idft = idft(img_dft_filtered)
    # plt_name = ['Original', 'Morph open/close', 'DFT', 'Binary w/ OSTU', 'Cut off max A(theta)', 'Output']
    # f, axs = plt.subplots(2,3)
    # f.set_figheight(10)
    # f.set_figwidth(70)
    # axs[0,0].imshow(img_src)
    # axs[0,0].set_title(plt_name[0])
    # axs[0,1].imshow(morph)
    # axs[0,1].set_title(plt_name[1])
    # axs[0,2].imshow(img_dft_log)
    # axs[0,2].set_title(plt_name[2])
    # axs[1,0].imshow(ostu)
    # axs[1,0].set_title(plt_name[3])
    # axs[1,1].imshow(img_dft_filtered_log)
    # axs[1,1].set_title(plt_name[4])
    # axs[1,2].imshow(img_idft)
    # axs[1,2].set_title(plt_name[5])
    # plt.show()

    # =========== test region ===========
    # image_path = 'data/20201229/EXT/resize/20201229080615_0EXT.bmp'
    # image_path = 'data/20201229/EXT/resize'
    # dft_denoiser(path=image_path, des_path='data/20201229/EXT/dft')