import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import os

def shiftedColorMap(cmap, start=0.0, midpoint=0.75, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mcolors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

label_file = ""
data_file = ""

exportDir = "" #figure export files

xyGene = np.genfromtxt(label_file, delimiter=' ', dtype=str, usecols=(0,))
xyData = np.genfromtxt(data_file, delimiter=' ', dtype=float, usecols=(0,1))

xyGeneDict = dict()
for i in range(len(xyGene)):
    xyGeneDict[xyGene[i]] = xyData[i,:]

####each file

fileList = list()
expression_data_files_dir = ""
for filename in os.listdir("expression_data_files_dir"):
    if not filename.endswith("specific_genes.txt"):
        continue
    rpkmFile = Path("expression_data_files_dir").joinpath(filename)
    rpkmGene = np.genfromtxt(rpkmFile, delimiter='\t', dtype=str, usecols=(0,), skip_header=1)
    rpkmData = np.genfromtxt(rpkmFile, delimiter='\t', dtype=float, usecols=(1,), skip_header=1)

    rpkmDataDict = dict()
    geneList = list()
    for i in range(len(rpkmGene)):
        if rpkmGene[i] not in xyGeneDict:
            continue
        if rpkmData[i] >= 4:
            rpkmData[i] = 4
        elif rpkmData[i] <= -1:
            rpkmData[i] = -1
        geneList.append(rpkmGene[i])
        rpkmDataDict[rpkmGene[i]] = rpkmData[i]

    rpkmXY = np.ndarray(shape=(len(geneList),2), dtype=float)
    rpkmValue = np.ndarray(shape=(len(geneList),1), dtype=float)

    index = 0
    for gene in geneList:
        rpkmXY[index] = xyGeneDict[gene]
        rpkmValue[index] = rpkmDataDict[gene]
        index += 1
    minValue = np.asscalar(min(rpkmValue))
    maxValue = np.asscalar(max(rpkmValue))
    print(minValue)
    print(maxValue)
    middleValue = (minValue + maxValue)/ 2.0
    midpoint = 0.5
    print(midpoint)

    orig_cmap = plt.get_cmap('coolwarm')
    shrunk_cmap = shiftedColorMap(orig_cmap, start=0.375, midpoint=midpoint, stop=1, name='shrunk')
    #
    # minStr = str(round(minValue,2))
    # maxStr = str(round(maxValue,2))
    # middleStr = str(round(middleValue,2))

    plt.figure(figsize=(80,50))
    plt.scatter(xyData[:,0], xyData[:,1], s=50, color = "silver")
    plt.scatter(rpkmXY[:,0], rpkmXY[:,1], s=150, c=rpkmValue.ravel(), cmap=shrunk_cmap)
    plt.axis("off")
    plt.colorbar().ax.tick_params(labelsize=50)
    plt.savefig(exportDir+filename.replace(".txt",".png"), bbox_inches='tight')

    plt.close()



