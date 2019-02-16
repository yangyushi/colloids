from matplotlib import pyplot as plt
from colloids import track
from colloids.particles import get_bonds
import numpy as np


def draw_circles(xs, ys, rs, **kwargs):
    for x, y, r in zip(xs, ys, rs):
        circle = plt.Circle((x, y), radius=r, **kwargs)
        plt.gca().add_patch(circle)


im = plt.imread('droplets.jpg')
finder = track.MultiscaleBlobFinder(im.shape, Octave0=False, nbOctaves=5, nbLayers=8)
centers = finder(im, k=1.2, maxedge=-1)
centers = centers[(centers[:, -2] > 4) & (centers[:, -1] < -0.5)]

s = track.radius2sigma(centers[:, -2], dim=2)
bonds, dists = get_bonds(positions=centers[:, :-2], radii=centers[:, -2], maxdist=3.0)

draw_circles(centers[:, 0], centers[:, 1], centers[:, -2], facecolor='none', edgecolor='g')
plt.imshow(im, 'hot')
plt.show()

brights1 = track.solve_intensities(s, bonds, dists, centers[:, -1])
radii1 = track.global_rescale_intensity(s, bonds, dists, brights1)

draw_circles(centers[:, 0], centers[:, 1], radii1, facecolor='none', edgecolor='g')
plt.imshow(im, 'hot')
plt.show()

brights2 = track.solve_intensities(s, bonds, dists, centers[:, -1], R0=radii1)
radii2 = track.global_rescale_intensity(s, bonds, dists, brights2, R0=radii1)

draw_circles(centers[:, 0], centers[:, 1], radii2, facecolor='none', edgecolor='g')
plt.imshow(im, 'hot')
plt.show()

histR0, bins = np.histogram(centers[:, -2], bins=np.arange(30))
histR1, bins = np.histogram(radii1, bins=np.arange(30))
histR2, bins = np.histogram(radii2, bins=np.arange(30))
plt.step(bins[:-1], histR0, label='supposed dilute')
plt.step(bins[:-1], histR1, label='corrected for overlap')
plt.step(bins[:-1], histR2, label='2nd iteration')
plt.xlabel('radius (px)')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()
