import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc,ndimage
from PIL import Image
import imageio
#from skimage import transform,io
import time


# Trabajo con Matplotlib e imagenes.

# Entrada, salida y presentación de imágenes
# Manipulación básica: recortar, girar, rotar, ...
# Filtrado de imágenes: reducción de ruido, enfoque/refinamiento (sharpening)
# Segmentación de imágenes: etiquetado de píxeles de acuerdo a los diferentes objetos a que puedan pertenecer
# Clasificación
# Extracción/identificación de patrones
# Registro


class Imagen():
    def __init__(self):
        face = imageio.imread('imagen.png')
        #face = Image.open('imagen.png').convert('LA')
        face = np.asarray(face)
        print(type(face))
        plt.imshow(face)
        plt.show()
    
    def recortarImagen(self):
        face = imageio.imread('imagen.png')
        face = np.asarray(face)
        print(type(face))
        print(face.shape)
        lx, ly = face.shape[0], face.shape[1]
        
        X, Y = np.ogrid[0:lx, 0:ly]
        mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
        face[mask] = 0
        # Cambiamos los bits de la imagen
        # Para hacer el efecto del circulo
        face[range(400), range(400)] = 255
        # Mostramos la iamgen modificada
        plt.imshow(face)
        plt.show()

    def girarImagen(self):
        # Rotación de la imagen
        face = imageio.imread('imagen.png')
        face = np.asarray(face)
        rotate_img_90 = ndimage.rotate(face, 90)
        rotate_img_45 = ndimage.rotate(face, 45)
        plt.imshow(rotate_img_90)
        plt.show()

    def rotarImagen(self):
        # Rotación de la imagen
        face = imageio.imread('imagen.png')
        face = np.asarray(face)
        rotate_img_90 = ndimage.rotate(face, 90)
        rotate_img_45 = ndimage.rotate(face, 45)
        plt.imshow(rotate_img_45)
        plt.show()

    def enfoque_refinamiento(self):
        img = imageio.imread('imagen.png')
        img = np.asarray(img)
        blurred_l = ndimage.gaussian_filter(img, 1.5)

        filter_blurred_l = ndimage.gaussian_filter(blurred_l, 0.5)

        alpha = 10
        sharpened = blurred_l + alpha * (blurred_l - filter_blurred_l)

        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(blurred_l, cmap=plt.cm.gray)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(sharpened, cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()

    def reducir_ruido(self):
        #l = scipy.misc.lena()
        # read in grey-scale
        #grey = io.imread('imagen.png', as_grey=True)
        # resize to 28x28
        # small_grey = transform.resize(grey, (28,28), mode='symmetric', preserve_range=True)
        # reshape to (1,784)
        #reshape_img = small_grey.reshape(1, 784)
        l = imageio.imread('imagen.png')
        l = np.asarray(l)
        l = l[:,:,0]
        #l = np.asarray(reshape_img)
        l = l[230:290,220:320]
        noisy = l + 0.1*l.std()*np.random.random(l.shape)
        print(noisy.shape)

        gauss_denoised = ndimage.gaussian_filter(noisy, 2)
        med_denoised = ndimage.median_filter(noisy, 3)


        plt.figure(figsize=(12,2.8))

        plt.subplot(131)
        plt.imshow(noisy,cmap=plt.cm.gray, vmin=40, vmax=220 )
        plt.axis('off')
        plt.title('noisy', fontsize=20)
        plt.subplot(132)
        plt.imshow(gauss_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
        plt.axis('off')
        plt.title('Gaussian filter', fontsize=20)
        plt.subplot(133)
        plt.imshow(med_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
        plt.axis('off')
        plt.title('Median filter', fontsize=20)

        plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                            right=1)
        plt.show()


    def extraccion_patrones_borde(self):
        im = imageio.imread('imagen.png')
        im.flatten()
        im = np.asarray(im,dtype='float64')
        #arr = np.add(arr, image.flatten(), out=arr, casting="unsafe")
        im = im[:,:,0]
        #im = np.zeros((256, 256))
        #im[64:-64, 64:-64] = 1

        im = ndimage.rotate(im, 15, mode='constant')
        im = ndimage.gaussian_filter(im, 8)

        sx = ndimage.sobel(im, axis=0, mode='constant')
        sy = ndimage.sobel(im, axis=1, mode='constant')
        sob = np.hypot(sx, sy)

        plt.figure(figsize=(16, 5))
        plt.subplot(141)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('square', fontsize=20)
        plt.subplot(142)
        plt.imshow(sx)
        plt.axis('off')
        plt.title('Sobel (x direction)', fontsize=20)
        plt.subplot(143)
        plt.imshow(sob)
        plt.axis('off')
        plt.title('Sobel filter', fontsize=20)

        im += 0.07*np.random.random(im.shape)

        sx = ndimage.sobel(im, axis=0, mode='constant')
        sy = ndimage.sobel(im, axis=1, mode='constant')
        sob = np.hypot(sx, sy)

        plt.subplot(144)
        plt.imshow(sob)
        plt.axis('off')
        plt.title('Sobel for noisy image', fontsize=20)



        plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=0.9)

        plt.show()


img_1 = Imagen()
img_1.recortarImagen()
img_1.girarImagen()
img_1.rotarImagen()
img_1.enfoque_refinamiento()
img_1.reducir_ruido()
img_1.extraccion_patrones_borde()