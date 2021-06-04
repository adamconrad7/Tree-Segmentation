import matplotlib.pyplot as plt
from tifffile import imread

from Modules.treeseg import *



path = "data/Mac_1120_UTM.tif"
# plt.rcParams['figure.dpi'] = 200
rgb = imread(path)
im = (rgb/65535).astype('float32')

labeler = Train(im, "coords.json", 30)
labeler.verify_chunks()
# labeler.label()

# m = Model("data/Mac_1120_UTM.tif", "coords.json", 30)
#
# m.check_data()
