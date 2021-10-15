import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_laplace, sobel
from sklearn import cluster
from sklearn.cluster import KMeans


def mykmeans(imgPath, imgFilename,
                              savedImgPath, savedImgFilename, k):
    """
            parameters:
            imgPath: the path of the image folder. Please use relative path
            imgFilename: the name of the image file
            savedImgPath: the path of the folder you will save the image
            savedImgFilename: the name of the output image
            k: the number of clusters of the k-means function 
            function: using k-means to segment the image and save the result to an image with a bounding box
    """
    # load the image data into memory, for future use we will load the image in all common formats
    img_org, img_rgb, img_gray, img_binary = reader(imgPath, imgFilename)

    # convert RGB to HSV in order to equalize the histogram to enhance brightness contrast
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # split HSV into separate channels
    chans = cv2.split(hsv)
    # use the second channel to filter out the bright background first
    chan = np.copy(chans[1])
    hsv[chan <= 15] = 0
    # equalize the image again to enhance the contrast among colors without background
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    # convert back to RGB format and we will process in RGB
    img_rgb_ = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # split the RGB channels
    R, G, _ = cv2.split(img_rgb_)
    # create a color mask which helps identify the difference between face color and light shirt color
    _mask = (R-G) < 25
    img_rgb_[_mask] = 0

    # filter out the noise left after preprocessing so far
    img_rgb_ = cv2.medianBlur(img_rgb_, ksize=7)

    # create a new pixel matrix with the same width and height as the original image
    # except that we add two more features in addition to RGB channels
    # thus, the fourth feature is the x's coordinate and fifth is the y's coordinate of each pixel
    shapes = img_rgb_.shape
    pix_ = np.zeros((shapes[0], shapes[1], 5))
    # loop through each pixel and add x, y coordinate behind RGB values
    # this step improve Kmean algorithm by not only clustering on the color
    # but also putting pixels into correct classes based on their location
    # this helps remedy the error of clustering faces and hands together
    # due to very similar RBG values
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            pix_[i, j] = np.concatenate((img_rgb_[i, j], [i, j]), axis=None)
    pix_ = pix_.reshape((-1, 5))

    # run Kmean algorithm with color features RGB and coordinates x,y
    kmean_ = KMeans(n_clusters=k, random_state=0).fit(pix_)
    labels = kmean_.labels_
    centers = kmean_.cluster_centers_

    # use the colorCenter method to find the list of centers that can represent the color of face
    cluster_ = colorCenter(centers)

    # use the mask method to filter out any irrelevant color clusters
    _, binary, _ = mask(img_rgb_, cluster_, labels)

    # as required in the instruction, this is just to showcase the intermediate outcome in binary
    plt.imshow(binary, cmap='gray')
    plt.show()

    # find the contour of the face in the preprocessed binary image
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all the contours
    cv2.drawContours(binary, contours, -1, (255, 0, 0), 1)

    # Iterate through all the contours
    for contour in contours:
        # Find bounding rectangles
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the rectangle
        if cv2.contourArea(contour) > 200:
            img_rgb = cv2.rectangle(
                img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # save the outcome image into the specified file
    writer(img_rgb, savedImgPath, savedImgFilename)
    # present the outcome image with red rectangles around the figures' faces
    plt.imshow(img_rgb)
    plt.show()


def mask(img, clusters, labels):
    """this method uses the clusters and labels matrix to segment the img

    Args:
        img (array): image array
        clusters (list): a list of clusters that represent the color of face
        labels (array): the image array in the form of k clusters

    Returns:
        arrays: return three image arrays in orginal RGB format, binary and grayscale formats
    """
    # save important properties of the image before making any changes
    img_shape = img.shape
    masked = np.copy(img)
    # reshape the image array
    masked = masked.reshape((-1, 3))
    # loop through each label and if the label is not any of the ideal cluster
    # we'd like to black the pixel out
    for i in range(len(labels.flatten())):
        if labels.flatten()[i] not in clusters:
            masked[i] = [0, 0, 0]
    # reshape the masked image so we can view it as RGB
    masked = masked.reshape(img_shape)
    # convert the image from RGB to grayscale and binary and plot them
    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 157, 255, cv2.THRESH_BINARY)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(masked)
    ax[0].set_title("RGB")
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title("Binary")
    ax[2].imshow(gray, cmap='gray')
    ax[2].set_title("Gray")
    plt.show()
    return masked, binary, gray


def writer(img, savedImgPath, savedImgFilename):
    """call this method to write the image matrix into a readable file

    Args:
        savedImgFilename (string): a string of the file name
        img (array): all pixel values of the image
    """
    # create the saved file path
    path = os.path.join(savedImgPath, savedImgFilename)
    # write the image into the file
    cv2.imwrite(filename=path,
                img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def reader(imgPath, imgFilename):
    """read the img from the path provided in the parameters and return three pixel matrices

    Args:
        imgPath (string): the path of the file location
        imgFilename (string): the name of the file in the directory

    Returns:
        tuple of arrays: orignal image object loaded by imread(), RGB array, grayscale array, binary array
    """
    # Create the file path of the target image
    path = os.path.join(imgPath, imgFilename)
    # Read the image into memory, we read all three types of image for future use, including binary, gray, rgb
    img_org = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return img_org, img_rgb, img_gray, img_binary


def colorCenter(centers):
    """this method helps find the ideal color clusters that can represent face

    Args:
        centers (array): a list of RGB values

    Returns:
        list: a list of ideal clusters that may contain face
    """
    # format the centers array into a list of RGB colortuples
    centers = np.uint8(centers)
    colors = [tuple(x) for x in centers]
    # define a color range that can represent the face color with lower bound in the first tuple and upper bound in the second
    color_range = [(220, 150, 180), (250, 250, 255)]
    # loop through each color tuple in the tuples list and if the color is within the color range, then return the cluster index
    clusters = []
    for i, color in enumerate(colors):
        if color >= color_range[0] and color <= color_range[1]:
            clusters.append(i)
    return clusters


if __name__ == "__main__":
    imgPath = "./"  # Write your own path
    imgFilename = "faces.jpg"
    savedImgPath = r''  # Write your own path
    savedImgFilename = "faces_extra_credit.jpg"
    k = 40
    mykmeans(imgPath, imgFilename,
                              savedImgPath, savedImgFilename, k)
