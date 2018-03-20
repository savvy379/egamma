#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for visualising the contents for the `image`-formatted data files."""

# Basic import(s)
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Main function definition.
def main ():
    
    # Define variable(s)
    paths = sys.argv[1:]

    # Check(s)
    if len(paths) == 0:
        print "Please specify at least one data file to inspect:"
        print " $ python {} <path/to/file> [...]".format(sys.argv[0])
        return

    bin_width = [
        (0.025,     0.1),         # PS
        (0.003125,  0.0245 * 4),  # ECAL barrel layer 1
        (0.025,     0.0245),      # ...           ... 2
        (0.025 * 2, 0.0245 * 2),  # ...
        ]
    title = {
        'image_layer0': "Presampler",
        'image_layer1': "ECAL Barrel 1",
        'image_layer2': "ECAL Barrel 2",
        'image_layer3': "ECAL Barrel 3",
        }

    # Loop data files
    stop = False
    for ipath, path in enumerate(paths):
        print "[{}/{}] Inspecting {}".format(ipath + 1, len(paths), path)
        with h5py.File(path, 'r') as hf:
            dataset = hf['egamma']
            for isample, sample in enumerate(dataset):
                features_img = filter(lambda n: 'image' in n, sample.dtype.names)

                # Get maximal cell energy for current sample
                vmax = max([sample[feat].max() for feat in features_img])
                norm = LogNorm(vmin=1E+00, vmax=vmax)

                # Canvas
                fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)

                # Plot all calorimeter image(s)
                dim = 2
                for ifeat, feat in enumerate(features_img):
                    
                    # Define current axes, image, bin widths
                    row, col = ifeat // dim, ifeat % dim
                    ax = axes[row, col]
                    im = sample[feat]
                    bw = bin_width[ifeat]

                    # Define axis arrays
                    num_eta, num_phi = im.shape
                    eta = np.linspace(- num_eta // 2 * bw[0], + num_eta // 2 * bw[0], num_eta + 1, endpoint=True)
                    phi = np.linspace(- num_phi // 2 * bw[1], + num_phi // 2 * bw[1], num_phi + 1, endpoint=True)

                    # Get mesh grid
                    Eta, Phi = np.meshgrid(eta,phi)
                    
                    # Plot images
                    pcol = ax.pcolor(Eta, Phi, im.T, cmap='Blues', norm=norm)

                    # Decorations
                    ax.set_title(title[feat])
                    ax.set_xlim(-0.2, 0.2)
                    ax.set_ylim(-0.2, 0.2)
                    if row == 1:
                        ax.set_xlabel(r"$\Delta\eta$")
                        pass
                    if col == 0:
                        ax.set_ylabel(r"$\Delta\phi$")
                        pass
                    ax.set_xticks(np.linspace(-0.2, 0.2, 5, endpoint=True))
                    ax.set_yticks(np.linspace(-0.2, 0.2, 5, endpoint=True))
                    pass

                # Decorations
                cbar = fig.colorbar(pcol, ax=axes.ravel().tolist())
                cbar.set_label("Energy [MeV]", rotation=270, labelpad=+20)

                # Save/show
                plt.savefig('tmp_images.pdf')
                #plt.show()

                print "Inspect next electron candidate ([y]), go to next file (n), or quit (q)? >>",
                response = raw_input('')
                if   response == 'n':
                    print "Going to next file."
                    break
                elif response == 'q':
                    print "Exiting."
                    stop = True
                    break
                pass
            pass

        # Manual break condition
        if stop: 
            break
        pass
    

    return 0


# Main function call.
if __name__ == '__main__':
    main()
    pass

