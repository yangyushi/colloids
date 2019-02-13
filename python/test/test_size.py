from matplotlib import pyplot as plt
from colloids import track
from colloids.particles import get_bonds
import numpy as np


def draw_circles(xs, ys, rs, **kwargs):
    for x, y, r in zip(xs, ys, rs):
        circle = plt.Circle((x, y), radius=r, **kwargs)
        plt.gca().add_patch(circle)


im = plt.imread('droplets.jpg')
finder = track.MultiscaleBlobFinder(im.shape, Octave0=False, nbOctaves=4)
centers = finder(im, k=1.6)

histR0, bins = np.histogram(centers[:,-2], bins=np.arange(30))

s = track.radius2sigma(centers[:, -2], dim=2)
bonds, dists = get_bonds(positions=centers[:, ::2], radii=centers[:, -2], maxdist=3.0)

radii1 = track.global_rescale_intensity(s, bonds, dists, centers[:, -1])
draw_circles(centers[:, 0], centers[:, 1], radii1, facecolor='none', edgecolor='g')
plt.imshow(im, 'hot')
plt.show()

histR1, bins = np.histogram(radii1, bins=np.arange(30))
plt.step(bins[:-1], histR0, label='supposed dilute')
plt.step(bins[:-1], histR1, label='corrected for overlap')
plt.xlabel('radius (px)')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()
