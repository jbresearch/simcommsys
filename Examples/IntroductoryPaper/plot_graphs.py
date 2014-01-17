#!/usr/bin/python
#
# Copyright (c) 2010 Johann A. Briffa
#
# This file is part of SimCommSys.
#
# SimCommSys is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SimCommSys is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
#
# Graph plotting script

import sys
import simcommsys as sc
import matplotlib.pyplot as plt
import numpy as np

# sets up the plots. Note that typeid indicates BER (1), SER(2) or FER(3)
def plot_all(typeid,compare=True):
   # create new figure
   plt.figure()
   # set up
   xscale = 'linear'
   showiter = False
   showtol = False
   limit = False

   if compare and typeid==2:
      ## Plot theoretical results

      # Uncoded transmission using BPSK over AWGN
      p = np.arange(0,10,0.1)
      ebno = np.power(10.0, p/10.0)
      ber = sc.qfunc(np.sqrt(2 * ebno))
      plt.plot(p,ber,'b.-',markevery=5, \
         label=r'Uncoded (theoretical)')

      ## Plot results from handbook

      # Unconcatenated Convolutional K=7 133,171
      p = [0.5, 1.03, 1.59, 2.02, 2.41, 2.69, 3.13, 3.52, 3.86, 4.25]
      ber = [0.0979, 0.0471, 0.0164, 0.00655, 0.0025, 0.00114, 0.00034, 0.000109, 3.53e-05, 1.03e-05]
      plt.plot(p,ber,'b+-', \
         label=r'Unconcatenated (NASA)')

      # Concatenated RS (255,233) + Convolutional K=7 133,171
      p = [1.54, 1.75, 1.9, 2.1, 2.23, 2.31]
      ber = [0.0435, 0.0222, 0.00984, 0.000911, 0.000103, 1.01e-05]
      plt.plot(p,ber,'bx-', \
         label=r'Concatenated (NASA)')

   # Plot our sim results

   sc.plotresults('Results/sim.jab.errors_hamming-random-awgn-bpsk-uncoded.txt', \
      typeid,xscale,showiter,showtol,'k.-',5, \
      label=r'Uncoded')

   sc.plotresults('Results/sim.jab.errors_hamming-random-awgn-bpsk-nrcc_133_171.txt', \
      typeid,xscale,showiter,showtol,'k+-',limit, \
      label=r'Unconcatenated')

   sc.plotresults('Results/sim.jab.errors_hamming-random-awgn-bpsk-concantenated-reedsolomon_255_223_gf256-map_interleaved-nrcc_133_171.txt', \
      typeid,xscale,showiter,showtol,'kx-',limit, \
      label=r'Concatenated')

   # Plot requirements
   #if compare and typeid==2:
   #   plt.plot([0,10], [5e-3,5e-3], 'r--') # uncompressed data
   #   plt.plot([0,10], [5e-6,5e-6], 'r-.') # compressed data

   # Ancillary
   # Specify the title and x and y labels if required
   # Note Tex notation is allowed
   plt.title(r'Performance of NASA Voyager Codes')
   plt.xlabel(r'Signal-to-Noise Ratio (per information bit) in dB')
   if typeid==2:
      plt.ylabel(r'Bit Error Rate')
   sc.xlim(1,10)
   if typeid==2:
      sc.ylim(1e-5,1e-1)

   # Specify where the legend should appear: upper/lower left/right/center or best
   plt.legend(labelspacing=0.15, loc='best')
   plt.grid()
   sc.typeset()
   return
#end of plotset

# Main body of code
def main():
   # find out if user wants us to show plot or not
   showplot = True
   if len(sys.argv) >= 2 and sys.argv[1] == 'noshow':
      showplot = False

   # do the plots
   plot_all(2)
   plt.savefig('Figures/nasa-all-ber.pdf')
   plot_all(10)
   plt.savefig('Figures/nasa-all-cpu.pdf')
   plot_all(2,False)
   plt.savefig('Figures/nasa-ber.pdf')

   # Display all figures
   if showplot:
      plt.show()
   return

# main entry point
# Run main, if called from the command line
if __name__ == '__main__':
   main()
