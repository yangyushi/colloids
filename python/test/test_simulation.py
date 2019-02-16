from matplotlib import pyplot as plt
from colloids import track
from colloids.particles import get_bonds
import numpy as np


def draw_circles(centres, z_chosen, **kwargs):
    for (x, y, z, r) in centres[:, :4]:
        dz = np.abs(z - z_chosen)
        if dz < r:
            r_real = np.sqrt(r**2 - dz**2)
            circle = plt.Circle((x, y), radius=r_real, **kwargs)
            plt.gca().add_patch(circle)


im = np.load('hard_sphere_volume.npy')
im = np.moveaxis(im, -1, 0)
finder = track.MultiscaleBlobFinder(im.shape, Octave0=False, nbOctaves=10, nbLayers=5)

centers = finder(im, k=1.7)
print(centers.shape)
im = np.moveaxis(im, 0, -1)

z = 30

draw_circles(centers, z, facecolor='none', edgecolor='teal', lw=4)
plt.imshow(im[:, :, z], 'hot')
plt.show()

s = track.radius2sigma(centers[:, -2], dim=2)
bonds, dists = get_bonds(positions=centers[:, :-2], radii=centers[:, -2], maxdist=8.0)

brights1 = track.solve_intensities(s, bonds, dists, centers[:, -1])
radii1 = track.global_rescale_intensity(s, bonds, dists, brights1)

draw_circles(np.hstack([centers[:, :3], np.array([radii1]).T]), z, facecolor='none', edgecolor='teal', lw=4)
plt.imshow(im[:, :, z], 'hot')
plt.show()

brights2 = track.solve_intensities(s, bonds, dists, centers[:, -1], R0=radii1)
radii2 = track.global_rescale_intensity(s, bonds, dists, brights2, R0=radii1)

draw_circles(np.hstack([centers[:, :3], np.array([radii2]).T]), z, facecolor='none', edgecolor='teal', lw=4)
plt.imshow(im[:, :, z], 'hot')
plt.show()

#histR0, bins = np.histogram(centers[:, -2], bins=np.arange(30))
#histR1, bins = np.histogram(radii1, bins=np.arange(30))
#histR2, bins = np.histogram(radii2, bins=np.arange(30))
#plt.step(bins[:-1], histR0, label='supposed dilute')
#plt.step(bins[:-1], histR1, label='corrected for overlap')
#plt.step(bins[:-1], histR2, label='2nd iteration')
#plt.xlabel('radius (px)')
#plt.ylabel('count')
#plt.legend(loc='upper right')
plt.show()
