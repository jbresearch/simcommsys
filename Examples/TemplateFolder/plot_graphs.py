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

# sets up the plots. Note that type indicates BER (1), SER(2) or FER(3)
def plotset(type):
   # create new figure
   plt.figure()
   # set up
   xscale = 'log' # possible values are 'log' (base 10) or 'linear'
   showiter = False
   showtol = False
   limit = False

   # Example for generating the graph of your result
   sc.plotresults('Results/name_of_your_sim_results.txt', \
      type,xscale,showiter,showtol,'k+-',limit, \
      label=r'NameOfGraph')
   # repeat the above line as needed to include other graphs

   # plot graphs with FER only
   if type==3:
      # Example for results specified manually
      p = [0.0015, 0.00176, 0.00201, 0.00225, 0.00251, 0.00301, 0.00354, 0.004, 0.00503];
      fer = [0.00015, 0.000244, 0.000426, 0.000609, 0.000971, 0.00302, 0.013, 0.033, 0.207];
      plt.plot(p,fer,'k*-', \
      label=r'NameOfGraph')
      # This can be useful if we only have published graphs and need to read off
      # the data points from the published image. That way we can show how our
      # results compare to any published results.

   # Ancillary
   # Specify the title and x and y labels if required
   # Note Tex notation is allowed
   plt.title(r'TitleOfGraph with Maths notation $(1,2)$')
   plt.xlabel(r'SomeDescription')
   sc.xlim(0.001,0.2)
   # Example how to set y-range depending on the type of graph (SER/FER/etc)
   if type==1:
      sc.ylim(1e-6,1)
   elif type==2:
      sc.ylim(1e-5,1)
   else:
      sc.ylim(1e-4,1)

   # Specify where the legend should appear: upper/lower left/right/center or best
   plt.legend(loc='upper left')
   # Example for legend outside axes:
   # sc.shrink_axes(0.8,0.95,0.0,0.05)
   # plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0., loc='upper left')
   # Finally show the grid and set fonts etc.
   plt.grid()
   sc.typeset()
   return
#end of plotset

# main body of code
def main():
   # find out if user wants us to show plot or not
   showplot = True
   if len(sys.argv) >= 2 and sys.argv[1] == 'noshow':
      showplot = False

   # do the plots - delete as appropriate
   plotset(1)
   plt.savefig('Figures/NameOfGraph-ber.pdf')
   plotset(2)
   plt.savefig('Figures/NameOfGraph-ser.pdf')
   plotset(3)
   plt.savefig('Figures/NameOfGraph-fer.pdf')

   # Display all figures
   if showplot:
      plt.show()
   return

# main entry point
# Run main, if called from the command line
if __name__ == '__main__':
   main()
