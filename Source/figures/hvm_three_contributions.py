import numpy as np
import lorenzmie.theory.spherefield as sphf
import pylab as pl
import latex_options
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import os

def image_field(E, nx, ny): 
    return np.sum(np.real(E*np.conj(E)), axis=0).reshape(nx, ny)

def generate_fields(nx, ny):    
    # Particle choices.
    a_p = 1.0  # [um]
    n_p = 1.59 # [1]
    z_p = 85  # [pix] 
    n_m = 1.33 # [1]
    mpp = 0.135  # [um/pix]
    lamb = 0.447 # [um]
    
    # Generate the scattered field.
    rp = [0,0]
    x = np.tile(np.arange(nx, dtype = float), ny)
    y = np.repeat(np.arange(ny, dtype = float), nx)
    x -= float(nx)/2. + float(rp[0])
    y -= float(ny)/2. + float(rp[1])
    E_scat = sphf.spherefield(x, y, z_p, a_p, n_p, n_m = n_m, cartesian = True, mpp = mpp, 
                         lamb = lamb)


    # Generate the incident field.
    k = 2.0*np.pi/(lamb/np.real(n_m)/mpp)
    E_inc = np.zeros(E_scat.shape, dtype=complex)
    E_inc[0,:] += 1.0*np.exp( -1j * k * z_p)

    # Generate mixed field.
    E_mixed = 2*np.real(E_inc*np.conj(E_scat))
    
    return E_inc, E_mixed, E_scat

def final_image():
    nx, ny = 200, 200
    E_inc, E_mixed, E_scat = generate_fields(nx, ny)

    # Generate the final image.
    E_total = E_inc + E_scat
    image = image_field(E_total, nx, ny)

    fig, ax = pl.subplots()

    ax.imshow(image, cmap='gray', interpolation='gaussian')
    ax.axis('off')
    
    filename = os.path.join('../../Figures/', 'hvm_three_contributions_01.png')
    pl.savefig(filename, dpi=200)

def three_contributions():
    nx, ny = 200, 200
    E_inc, E_mixed, E_scat = generate_fields(nx, ny)

    # Generate the images of the three contributions.
    image_mixed = np.sum(E_mixed, axis=0).reshape(nx, ny)
    image_inc = image_field(E_inc, nx, ny)
    image_scat = image_field(E_scat, nx, ny)

    # Plot the images.
    fig, axes = pl.subplots(1, 3, figsize=pl.figaspect(0.33))
    for ax in axes:
        ax.axis('off')

    ax_einc = axes[0].imshow(image_inc, cmap='gray', interpolation='gaussian', vmin=0, vmax=1.5)
    ax_mixed = axes[1].imshow(image_mixed, cmap='gray', interpolation='gaussian')
    ax_scat = axes[2].imshow(image_scat, cmap='gray', interpolation='gaussian')
    #pl.show()
    
    filename = os.path.join('../../Figures/', 'hvm_three_contributions_02.png')
    pl.savefig(filename, dpi=200)
    
def three_contributions_3d():
    
    nx, ny = 200, 200
    E_inc, E_mixed, E_scat = generate_fields(nx, ny)
    
    # Generate the final image.
    E_total = E_inc + E_scat
    image = image_field(E_total, nx, ny)
    
    # Generate the images of the three contributions.

    image_mixed = np.sum(E_mixed, axis=0).reshape(nx, ny)
    image_inc = image_field(E_inc, nx, ny)
    image_scat = image_field(E_scat, nx, ny)

    # Plot Figure.
    X = np.arange(nx)
    Y = np.arange(ny)
    X, Y = np.meshgrid(X, Y)
    fig = pl.figure(figsize=pl.figaspect(0.33))
    cmap = 'plasma'
    vmin, vmax = -0.5, 1.0

    # Plot all three image surfaces.
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    surf1 = ax1.plot_surface(X, Y, image_inc, rstride=1, cstride=1, cmap=cmap,
                             linewidth=0, antialiased=False)
    
    # set up the axes for the second plot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    surf2 = ax2.plot_surface(X, Y, image_mixed, rstride=1, cstride=1, cmap=cmap,
                           linewidth=0, antialiased=False)

    
    # set up the axes for the second plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    surf3 = ax3.plot_surface(X, Y, image_scat, rstride=1, cstride=1, cmap=cmap,
                           linewidth=0, antialiased=False)

    # Fix views and limits.
    ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax1.set_zlabel('Intensity [arb]', rotation=90)
    ax1.zaxis.labelpad=10
    for ax in [ax1, ax2, ax3]:
        ax.set_zlim(-1.0, 1.2)
        ax.view_init(elev=7.5, azim=45)

        # Get rid of colored axes planes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.set_xlabel('X [pixel]')
        ax.set_ylabel('Y [pixel]')
        ax.yaxis.labelpad=10
        ax.xaxis.labelpad=10
        ax.set_xticks([0,100,200])
        ax.set_yticks([0,100])
        ax.set_zticks([-0.5, 0.0, 0.5, 1.0])
        
    filename = os.path.join('../../Figures/', 'hvm_three_contributions_03.png')
    pl.savefig(filename, dpi=200)
        
if __name__ == '__main__':
    final_image()
    three_contributions()
    three_contributions_3d()

    

    
