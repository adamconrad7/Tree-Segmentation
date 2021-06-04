import matplotlib.pyplot as plt
from tifffile import imread
from importlib import reload
import treeseg
path = 'data/Mac_1120_UTM.tif'
cpath = 'coords.json'
plt.rcParams['figure.dpi'] = 200


# Read in image and convert from 16-bit color depth
rgb = imread(path)
im = (rgb/65535).astype('float32')

# Instantiate Train class 
labeler = Train(im, cpath, 32)

# Verify that labeled data looks okay
labeler.verify_chunks()

# Instantiate Model class 
m = Model(im, cpath, 32)

# Train model with labeled data
m.train()

# Instantiate Segmenter class using trained model
s = Segmenter(m.im, cpath, 32, m.model)

# Get square regions of image to clasify over
regions = s.get_regions()

# Produce binary tiff for use in GIS
s.binary_im(regions)