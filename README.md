# Brain cancer experiments 

## Installation

The default paths assume that you have a `data-bwohlmuth` directory from Bene in the project directory.
If you don't know Bene, you are probably out of luck with executing anything. :(

Necessary packages:
- nibabel (`pip install nibabel`, or see https://nipy.org/nibabel/)
- dipy (`pip install dipy`, or see https://dipy.org/).
Optional packages
- fenics with cbc.blocks (if you want to run a simulation).

TODO: Add an index file for easy installations.

## Generating all diffusion tensors tensors

```
> python niitumorio.py --create-all 
```

## Loading data 

```
from niitumorio import load_nii, create_path
import matplotlib.pyplot as plt

mask_path = create_path(directory='data-bwohlmuth', data='tissuemask')
# we want the slice at z = 79 
data = load_nii(mask_path, offsets_left=(0,0,79), offsets_right=(None, None, 80))

plt.imshow(data)
plt.show()
```