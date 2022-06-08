import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def version_1_ep_30():
    fnt_dict = {
        'fontsize': 45
    }
    plt.figure(figsize=(30, 20))
    plt.subplot(1, 2, 1)
    plt.title('model prediction',fnt_dict)
    img = mpimg.imread('../resources/result/out_13.png')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('original photo',fnt_dict)
    img = mpimg.imread('D:/studia/sem_6/PSI/landscape Images/color/7007.jpg')
    plt.imshow(img)
    plt.show()

def version_1_ep_30_2_forrest():
    fnt_dict = {
        'fontsize': 45
    }
    plt.figure(figsize=(30, 20))
    plt.subplot(1, 2, 1)
    plt.title('model prediction',fnt_dict)
    img = mpimg.imread('../resources/result/out_69.png')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('original photo',fnt_dict)
    img = mpimg.imread('D:/studia/sem_6/PSI/landscape Images/color/7063.jpg')
    plt.imshow(img)
    plt.show()

def version_1_ep_30_2_forrest():
    fnt_dict = {
        'fontsize': 45
    }
    plt.figure(figsize=(30, 20))
    plt.subplot(1, 2, 1)
    plt.title('model prediction',fnt_dict)
    img = mpimg.imread('../resources/result/out_69.png')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('original photo',fnt_dict)
    img = mpimg.imread('D:/studia/sem_6/PSI/landscape Images/color/7063.jpg')
    plt.imshow(img)
    plt.show()


def version_1_ep_30_3_woman():
    fnt_dict = {
        'fontsize': 45
    }
    plt.figure(figsize=(30, 20))
    plt.subplot(1, 2, 1)
    plt.title('model prediction',fnt_dict)
    img = mpimg.imread('../resources/result/out_137.png')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('original photo',fnt_dict)
    img = mpimg.imread('D:/studia/sem_6/PSI/woman.jpg')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    version_1_ep_30_3_woman()
