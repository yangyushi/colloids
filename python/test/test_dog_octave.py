from matplotlib import pyplot as plt
from colloids import track


im = plt.imread('droplets.jpg')
finder = track.MultiscaleBlobFinder(im.shape, Octave0=False, nbOctaves=4)
centers = finder(im, maxedge=-1)

m = max([oc.layers.max() for oc in finder.octaves[1:]])
for o, oc in enumerate(finder.octaves[1:]):
    for l, lay in enumerate(-oc.layers):
        a = plt.subplot(len(finder.octaves) - 1, len(oc.layersG), len(oc.layersG) * o + l + 1)
        # hide ticks for clarity
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        plt.imshow(lay, 'hot', vmin=0, vmax=m)
plt.show()
