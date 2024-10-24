import os
from scipy import misc
from scipy.optimize import curve_fit
import PIL
import numpy as np
import matplotlib.pyplot as plt

# file_path = ['':'/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal1M3V_Light3_bitmap.bmp', '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal1M5V_Light0_Invert.bmp', '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal1M7V_Light3_bitmap.bmp', '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_25M_3V_bitmap.bmp', '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_0.25M_5V_bitmap.bmp']

path = '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal25M7V_light3_bitmap.bmp'
path = '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals0.01M5V_bitmap.bmp'
# image= misc.imread('/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal1M3V_Light3_bitmap.bmp')

image = PIL.Image.open(path)
print(image.format, image.size, image.mode)
# image.show()

image = np.array(image)
image = np.rot90(image, 3)
# for i in range(0, 200):
#     # print(image[i, i])


def linear_func(x, m, c):
    return m*x + c

def centre_of_mass(img):
    x, y = np.where(img == 255)
    x_com, y_com = np.mean(x), np.mean(y)
    return x_com, y_com


def mass_dimension(data_file):
    x, y = data_file[4], data_file[5]
    img = PIL.Image.open(data_file[6])
    img = np.array(img)
    img = np.rot90(img, 3)
    # x, y = int(centre_of_mass(img)[0]), int(centre_of_mass(img)[1])
    if x == 'centrex':
        # x = img.shape[0]//2
        x = int(centre_of_mass(img)[0])

    print(img.shape)

    if y == 'centrey':
        y = img.shape[1]//2
    else:
        y = img.shape[1] - y
        print(x)

    area_vals = []

    grid = np.meshgrid(np.arange(0, img.shape[0], 1), np.arange(0, img.shape[1], 1))
    # print(np.min([x,y]))
    radii = np.arange(0, np.min([x,y])+1, 1)
    for r in radii:
        sector = img[x-r:x+r+1, y-r:y+r+1]
        q = np.where(sector >= 100)

        Area = len(q[0])
        area_vals.append(Area)

    
    area_vals = np.array(area_vals)
    
    ln_area_vals, ln_radii = np.log(area_vals), np.log(radii+0.5)
    ln_r_bound = np.where((ln_radii >= data_file[2]) & (ln_radii <= data_file[3]))
    
    lin_params, lin_cov = curve_fit(linear_func, ln_radii[ln_r_bound], ln_area_vals[ln_r_bound])
    sigma = np.sqrt(np.diag(lin_cov))


    fig, axs = plt.subplots(2)
    fig.set_size_inches(10, 30)
    fig.suptitle(f'Molarity $= {data_file[0]}$, Voltage $= {data_file[1]}$')
    
    axs[0].set_title(f'Fractal: Molarity $= {data_file[0]}$, Voltage $= {data_file[1]}$')
    axs[0].plot(q[0], q[1], marker='s', color='r', markersize=0.5, linestyle='None')

    axs[1].set_title(f'Mass Dimension: Molarity $= {data_file[0]}$, Voltage $= {data_file[1]}$')
    axs[1].plot(ln_radii, ln_area_vals, 'b.')
    axs[1].plot(np.linspace(0, 6, 100), linear_func(np.linspace(0, 6, 100), *lin_params), 'r')
    axs[1].text(0,10, f'$D = {lin_params[0]:.2f} \pm {sigma[0]:.2f}$')
    axs[1].set_xlabel('$\ln(r)$')
    axs[1].set_ylabel('$\ln(N(r))$')

    plt.savefig
    plt.show()





data_files = [
['Molarity', 'Voltage','min_lnrad','max_lnrad', 'centre_x', 'centre_y', 'path'],
[      0.01,         5,       3.75,     np.inf, 'centrex' , 'centrey' , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals0.01M5V_bitmap.bmp'],
[      0.25,         3,        2.9,          5, 187       , 189       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_25M_3V_bitmap.bmp'],
[      0.25,         5,        2.5,        4.7, 285       , 265       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_0.25M_5V_bitmap.bmp'],
[      0.01,         7,        3.8,        5.2, 'centrex' , 'centrey' , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_0.01M_7V_bitmap.bmp'],
[      0.1 ,         7,       2.65,        4.4, 355       , 274       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_0.1M_7V_bitmap.bmp'],
[      0.1 ,         5,        2.8,        4.2, 295       , 149       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_0.1M_5V_bitmap.bmp'],
[      0.01,         3,        2.6,        4.6, 222       , 224       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_0.01M_3V_bitmap.bmp'],
[      0.1 ,         3,        2.6,        4.5, 156       , 159       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractals_0.1M_3V_bitmap.bmp'],
[      0.25,         7,        3.6,        4.9, 251       , 229       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal25M7V_light3_bitmap.bmp'],
[      1   ,         7,        2.2,        3.9, 255       , 275       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal1M7V_Light3_bitmap.bmp'],
[      1   ,         5,        2.5,          4, 242       , 180       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal1M5V_Light0_Invert.bmp'],
[      1   ,         3,        2.7,        3.7, 196       , 170       , '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Fractals/fractal images/Fractal1M3V_Light3_bitmap.bmp']
]        

for data in data_files[1:]:
    # image = PIL.Image.open(data[4])
    # # image.show()
    # print(image.size)
    # image = np.array(image)
    # # print(image.shape)
    # image = np.rot90(image, 3)
    # print(image.shape)
    # print('----------------')
    mass_dimension(data)
    
    