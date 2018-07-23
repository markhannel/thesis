import pylab as pl
import os
import string
import latex_options

def generate_fig_labels():
    # Produce the string.
    alpha = string.ascii_lowercase
    labels = r')        ('.join(alpha[0:10])
    labels = '(' + labels + ')'

    fig, ax = pl.subplots(figsize=(10,1))

    ax.text(0,0.5,labels, fontsize=24)
    ax.axis('off')
    
    #pl.show()
    filename = os.path.join('../../Figures/figure_labels.png')
    pl.savefig(filename, dpi=1000)

if __name__ == '__main__':
    generate_fig_labels()
