from matplotlib import pyplot as plt
from colloids import track


def draw_circles(xs, ys, rs, **kwargs):
    for x, y, r in zip(xs, ys, rs):
        circle = plt.Circle((x, y), radius=r, **kwargs)
        plt.gca().add_patch(circle)


im = plt.imread('droplets.jpg')
finder = track.MultiscaleBlobFinder(im.shape, Octave0=False, nbOctaves=4)
centers = finder(im, k=1.6)
draw_circles(centers[:, 0], centers[:, 1], centers[:, 2], facecolor='none', edgecolor='g')
plt.imshow(im, 'hot')
plt.show()
