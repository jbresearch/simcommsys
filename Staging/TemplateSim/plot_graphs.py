#!/usr/bin/python
#$Id$

import sys
import simcommsys as sc
import matplotlib.pyplot as plt
import numpy as np

# sets up the plots. Note that type indicates BER (1), SER(2) or FER(3)
def plotset(type):
   # create new figure
   plt.figure()
   # set up
   legendlist = []

   #specify the x-scale possible values are 'log' (base 10) or 'linear'
   xscale = 'log'

   showiter = False
   showtol = False
   limit = False

   #plot graphs with BER
   if type==1:
      #the following 2 lines will generate the graph of your result
      sc.plotresults('Results/name_of_your_sim_results.txt',type,xscale,showiter,showtol,'k+-',limit)
      legendlist.append(r'NameOfGraph')
      #repeat the above 2 lines as many  times as needed to include other graphs

   #plot graphs with SER
   if type==2:
      #the following 2 lines will generate the graph of your result
      sc.plotresults('Results/name_of_your_sim_results.txt',type,xscale,showiter,showtol,'k+-',limit)
      legendlist.append(r'NameOfGraph')
      #repeat the above 2 lines as many  times as needed to include other graphs

   #plot graphs with FER
   if type==3:
      #the following 2 lines will generate the graph of your result
      sc.plotresults('Results/name_of_your_sim_results.txt',type,xscale,showiter,showtol,'k+-',limit)
      legendlist.append(r'NameOfGraph')
      #repeat the above 2 lines as many  times as needed to include other graphs

      #Alternatively, you can specify manual data points as follows
      p = [0.0015, 0.00176, 0.00201, 0.00225, 0.00251, 0.00301, 0.00354, 0.004, 0.00503];
      fer = [0.00015, 0.000244, 0.000426, 0.000609, 0.000971, 0.00302, 0.013, 0.033, 0.207];
      plt.plot(p,fer,'k*-')
      legendlist.append(r'NameOfGraph')
      # This can be useful if we only have published graphs and need to read off
      # the data points from the published image. That way we can show how our results
      # compare to any published results.

   # Ancillary
   # Specify the title and x and y labels if required
   # Note Tex notation is allowed
   plt.title(r'TitleOfGraph with Maths notation $(1,2)$')
   plt.xlabel(r'SomeDescription')
   sc.xlim(0.001,0.2)

   #again depending on the type of graphs(BER, SER and FER) the
   # y-range might need to be set differently
   if type==1:
      sc.ylim(1e-6,1)
   elif type==2:
      sc.ylim(1e-5,1)
   else:
      sc.ylim(1e-4,1)

   #specify where the legend should appear: upper/lower left/right/center or best
   plt.legend(legendlist, loc='upper left')
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
