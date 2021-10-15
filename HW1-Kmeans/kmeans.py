# For students:
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def mykmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k):
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
    org, rgb, gray, binary = reader(imgPath, imgFilename)

    # run prepacked KMean method with specified K number of clusters
    labels, centers = kmeans(rgb, k)

    # use the colorCenter method to find the list of centers that can represent the color of face
    cluster_ = colorCenter(centers)

    # use the mask method to filter out any irrelevant color clusters
    _, binaryOut, _ = mask(
        rgb, clusters=cluster_, labels=labels)

    # apply median filtering algorithm and ksize to be 13 (empirical outcome)
    # this step is intended to remove insignificant white dots (salt-pepper problem)
    binary_ = cv2.medianBlur(binaryOut, ksize=13)

    # as required in the instruction, this is just to showcase the intermediate outcome in binary
    plt.imshow(binary_, cmap='gray')
    plt.show()

    # find the contour of the face in the preprocessed binary image
    contours, hierarchy = cv2.findContours(
        binary_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all the contours
    img = cv2.drawContours(binary_, contours, -1, (255, 0, 0), 1)

    # Iterate through all the contours
    for contour in contours:
        # Find bounding rectangles
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the rectangle
        rgb = cv2.rectangle(rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # save the outcome image into the specified file
    writer(rgb, savedImgPath, savedImgFilename)
    # present the outcome image with a red rectangle around the figure's face
    plt.imshow(rgb)
    plt.show()


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
    color_range = [(210, 160, 140), (220, 170, 150)]
    # loop through each color tuple in the tuples list and if the color is within the color range, then return the cluster index
    clusters = []
    for i, color in enumerate(colors):
        if color >= color_range[0] and color <= color_range[1]:
            clusters.append(i)
    return clusters


def kmeans(data, k, kind="rgb"):
    """wrap up the Kmean algorithm for RGB or Gray image

    Args:
        data (array): image matrix/array which we will treat as X input for Kmean
        k (int): the number of clusters we'd like to run the Kmean on
        kind (str, optional): specify the type of image data we input. Defaults to "rgb".

    Returns:
        arrays: labels is the array denoted by the k clusters and centers is 
                the array containing k color centers
    """
    # preprocess the input data based on the type of image data they are
    # the difference is that grayscale data only have one color channel,
    # whereas RGB have three
    if kind == "gray":
        pixel_values = data.reshape((1, data.shape[0]*data.shape[1]))
        pixel_values = np.float32(pixel_values)
    else:
        pixel_values = data.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
    # specify the parameters required by cv2.kmeans method
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # run kmeans on the preprocessed data and collect the outcomes and return
    compactness, labels, (centers) = cv2.kmeans(
        pixel_values, k, None, criteria=criteria,
        attempts=10, flags=flags
    )
    return labels, centers


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


if __name__ == "__main__":
    imgPath = "./"  # Write your own path
    imgFilename = "face_d2.jpg"
    savedImgPath = r''  # Write your own path
    savedImgFilename = "face_d2_face.jpg"
    k = 7
    mykmeans(imgPath, imgFilename,
                              savedImgPath, savedImgFilename, k)
