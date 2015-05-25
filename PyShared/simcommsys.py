#!/usr/bin/python
# coding=utf8
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
# Python library

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import xlrd
import math
import os
from scipy.integrate import quad

# Math / combinatorics functions

def qfunc(x):
   import scipy.special as sp
   return 0.5 * sp.erfc(x/np.sqrt(2.0))

def qfuncinv(x):
   import scipy.special as sp
   return np.sqrt(2) * sp.erfcinv(2*x)

def combinations(n,k):
   '''The number of k-combinations in a set of size n.'''
   if k > n:
      return 0
   return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def levenshtein_distance(first, second, wd=1, wi=1, ws=1):
   '''Find the Levenshtein distance between two strings.'''
   if len(first) > len(second):
      first, second = second, first
   if len(second) == 0:
      return len(first)
   first_length = len(first) + 1
   second_length = len(second) + 1
   distance_matrix = [[0] * second_length for x in range(first_length)]
   for i in range(first_length):
      distance_matrix[i][0] = i
   for j in range(second_length):
      distance_matrix[0][j]=j
   for i in xrange(1, first_length):
      for j in range(1, second_length):
         deletion = distance_matrix[i-1][j] + wd
         insertion = distance_matrix[i][j-1] + wi
         substitution = distance_matrix[i-1][j-1]
         if first[i-1] != second[j-1]:
            substitution += ws
         distance_matrix[i][j] = min(insertion, deletion, substitution)
   return distance_matrix[first_length-1][second_length-1]

def hamming_weight(x):
   w = 0
   while x > 0:
      w += (x & 1)
      x >>= 1
   return w

def average_weight(k):
   '''Returns: (w)

   Where:
      k = number of bits per symbol (assumes q = 2^k)

   Computes the average weight of all k-bit sequences.
   Since the sequence is complete, this should really be k/2.
   '''

   w = 0
   for i in xrange(1<<k):
      w += hamming_weight(i)
   w /= float(1<<k)
   return w

# Determine mutual information from std dev of binary LLR

def compute_information(sigma):
   def integrand(x, sigma):
      ssq = sigma**2
      return np.exp(-(x-ssq/2)**2/(2*ssq))/np.sqrt(2*np.pi*ssq)*np.log2(1+np.exp(-x))
   I = [1-quad(integrand, -100, 100, args=sigma)[0] for sigma in sigma]
   return np.array(I)

# Functions to determine drift range

def drift_prob_positive(x, N, Pi, Pd):
   assert x >= 0
   # set constants
   Pt = 1 - Pi - Pd
   # main computation
   Pr = 0
   for i in range(0,N+1):
      tmp = float(combinations(N+x+i-1, x+i))
      tmp *= float(combinations(N, i))
      tmp *= pow(Pi * Pd / Pt, i)
      Pr += tmp
   Pr *= pow(Pt, N) * pow(Pi, x)
   return Pr

def drift_prob_negative(x, N, Pi, Pd):
   assert x <= 0
   # set constants
   Pt = 1 - Pi - Pd
   # main computation
   Pr = 0
   for i in range(-x,N+1):
      tmp = float(combinations(N+x+i-1, x+i))
      tmp *= float(combinations(N, i))
      tmp *= pow(Pi * Pd / Pt, i)
      Pr += tmp
   Pr *= pow(Pt, N) * pow(Pi, x)
   return Pr

def drift_prob(x, N, Pi, Pd):
   if x < 0:
      return drift_prob_negative(x, N, Pi, Pd)
   return drift_prob_positive(x, N, Pi, Pd)

def find_max_drift(Pr, N, Pi, Pd):
   # determine area that needs to be covered
   Pr = 1 - Pr
   # determine xmax to use
   max = 0
   Pr -= drift_prob(max, N, Pi, Pd)
   while True:
      max += 1
      Pr -= drift_prob(max, N, Pi, Pd)
      Pr -= drift_prob(-max, N, Pi, Pd)
      if Pr <= 0:
         break
   return max

# Algorithm derived from Davey
def compute_max_drift(Pr, N, Pi, Pd):
   # this is a limitation of the algorithm
   assert Pi == Pd
   # determine xmax to use
   return int(math.ceil( qfuncinv(Pr/2) * math.sqrt(2 * N * Pd / (1-Pd)) ))

def compute_max_insertions(Pr, N, Pi):
   I = (np.ceil((np.log(Pr) - np.log(N)) / np.log(Pi))) - 1
   return np.maximum(I, 1)

# Functions to process/correct results

def correct_ser_to_ber(ser):
   '''Returns: (ber)

   Where:
      ser = symbol error rate (can be a vector)

   Converts the SER to an equivalent BER, assuming:
   * exactly 'k' (integral) bits are required to represent the symbols
   * the symbol error rate is symmetric across all pairs of symbols
   '''

   # it can be shown that the factor is equal to the average weight of all
   # k-bit representations divided by k. The former is easily shown to be
   # equal to k/2, so that the correction factor is a constant 1/2.

   ber = ser * 0.5
   return ber

def correct_bsid_to_normerror(par, Ps, Pd, Pi, Reff):
   '''Returns: (norm)

   Where:
      par = BSID parameter vector
      Ps = multiplier to obtain Pr(substitution)
      Pd = multiplier to obtain Pr(deletion)
      Pi = multiplier to obtain Pr(insertion event)
      Reff = effective code rate

   Converts the BSID parameter values to a normalized rate of channel
   error events per information bit.
   '''

   Ps *= par
   Pd *= par
   Pi *= par

   norm = (Ps + Pd + Pi/(1-Pi)) / Reff
   return norm

# Functions to handle results files

def loaddata(filename,latest=True):
   '''Returns: (data,comments)

   Where:
      filename = file to be loaded
      latest = if true (default) load only the latest simulation in file
               if false, concatenate all simulations in file
               if a list, concatenate the specified simulations in file
      data = matrix of data values
      comments = string comments in file, in order of appearance
   '''

   fid = open(filename,'r')
   if not fid:
      print 'Cannot open file "%s".\n' % filename
      return

   data = []
   comments = []
   # read all lines in the file
   i = 0 # line counter
   j = 0 # block counter
   this_data = []
   this_comments = []
   for line in fid:
      if len(line)>0 and line[0] == '#':
         # if this set of comments is a block divider
         if len(line)>1 and line[1] == '%' and this_data != []:
            # store existing block
            data.append(this_data)
            comments.append(this_comments)
            # reset for next block
            this_data = []
            this_comments = []
            # increment block counter
            j += 1
         if len(line)>1:
            this_comments.append(line.lstrip('#% ').rstrip())
      else:
         this_data.append([float(s) for s in line.split()])
      i += 1
   # store last block if necessary
   if this_data != []:
      data.append(this_data)
      comments.append(this_comments)
   # figure out what we need to return
   if isinstance(latest, list):
      this_data = sum([x for i,x in enumerate(data) if i in latest], [])
      this_comments = sum([x for i,x in enumerate(comments) if i in latest], [])
   elif latest:
      this_data = data.pop()
      this_comments = comments.pop()
   else:
      this_data = sum(data, [])
      this_comments = sum(comments, [])

   return (np.array(this_data),this_comments)

def loadresults(filename,latest=True):
   '''Returns: (par,results,tolerance,passes,cputime,header,comments)

   Where:
      filename = file to be loaded
      latest = if true (default) load only the latest simulation in file
               if false, concatenate all simulations in file
               if a list, concatenate the specified simulations in file

      par = column vector with parameter values simulated
      results = matrix of results *, **
      tolerance = matrix of result tolerance limits *, **
      passes = column vector with number of passes performed *
      cputime = column vector with CPU time (in seconds) required *
      header = cell array with description of corresponding results
      comments = column array of string comments in file, in order
                 of appearance

   *  each row's parameter is the corresponding row in 'par'
   ** columns corresponds to successive results issued by simulator.
      these can be:
         - BER,FER repeated for each successive iteration
         - BER,SER,FER repeated as above
         - SER for each successive symbol value, repeated as above
      Interpretation can be done using the 'header' field.
   '''

   # get data from file
   (data,comments) = loaddata(filename,latest)
   # if there is nothing, return quickly
   if not data.size:
      empty = np.array([])
      return (empty,empty,empty,empty,empty,[],[])

   # sort rows in ascending order of parameter value
   data = data[data[:,0].argsort(),]

   # find the header field if present, and convert to a list
   header = []
   for line in comments:
      if line.startswith('Par\t'):
         header = line.split('\t')

   # if we don't have a header (old-style), create labels manually
   if header == []:
      # number of data columns
      n = len(data[0])
      n = (n-2)/2
      # prefix
      header.append('Par')
      # main
      if n % 3 == 0:
         # assume we have BER,SER,FER triples
         for i in range(0,n,3):
            header.append('BER')
            header.append('Tol')
            header.append('SER')
            header.append('Tol')
            header.append('FER')
            header.append('Tol')
      else:
         # assume we have BER,FER duples
         for i in range(0,n,2):
            header.append('BER')
            header.append('Tol')
            header.append('FER')
            header.append('Tol')
      # suffix
      header.append('Samples')

   # extract the various pieces
   (par,data,header) = get_columns(data,header,'Par')
   (passes,data,header) = get_columns(data,header,'Samples')
   (cputime,data,header) = get_columns(data,header,'CPUtime')
   (tolerance,data,header) = get_columns(data,header,'Tol')
   results = data

   return (par,results,tolerance,passes,cputime,header,comments)

def get_columns(data,header,description,exact=False):
   '''Returns: (set,data,header)

   data = 2d array, each row a different experiment
   header = list of strings, one for each col of data
   description = text to find in header
   exact = true if we want exact matches for the whole word only

   Returns the set of data for each column starting with (or equal to, if
   requested) the given text. Also returns the updated data and header (with
   extracted data removed).'''

   # Find the matching positions
   if exact:
      i = [s == description for s in header]
   else:
      i = [s.startswith(description) for s in header]
   # Determine dimensions of final result
   n = i.count(True)
   m = len(data)
   # return early if we found nothing
   if n == 0:
      return (np.array([]),data,header)
   # Replicate over all rows
   ii = np.tile(i,(m,1))
   # Extract that set
   set = data[ii].copy().reshape(m,n)
   # Remove that set from the remaining data & header
   data = data[np.logical_not(ii)].copy().reshape(m,len(i)-n)
   header = np.array(header)[np.logical_not(i).nonzero()].tolist()
   return (set,data,header)

# set of styles, to be used in order
def get_styles(N1=None, N2=None, N3=None):
   # start with largest possible set
   symbols = [ '+', 'x', 'd', 'o', 's', '*', 'h', '^', 'v', '<', '>' ]
   ltypes = [ '--', '-', ':', '-.' ]
   colors = [ 'k', 'b', 'r', 'g', 'm', 'c', 'y' ]
   # reduce according to the number required
   if (N1):
      symbols = symbols[0:N1]
   if (N2):
      ltypes = ltypes[0:N2]
   if (N3):
      colors = colors[0:N3]
   # create combinations
   if (N1 and N2 and not N3):
      styles = [ (c,s,l) for (c,l) in zip(colors,ltypes) for s in symbols ]
   else:
      styles = [ (c,s,l) for c in colors for l in ltypes for s in symbols ]
   return styles

# set of styles, to be used in order
def get_styles2(N1=None, N2=None, N3=None):
   # start with largest possible set
   ltypes = [ '-', '--', ':', '-.' ]
   symbols = [ '+', 'x', 'd', 'o', 's', '*', 'h', '^', 'v', '<', '>' ]
   colors = [ 'k', 'b', 'r', 'g', 'm', 'c', 'y' ]
   # reduce according to the number required
   if (N1):
      ltypes = ltypes[0:N1]
   if (N2):
      symbols = symbols[0:N2]
   if (N3):
      colors = colors[0:N3]
   # create combinations
   if (N1 and N2 and not N3):
      styles = [ (c,s,l) for (c,l) in zip(colors,ltypes) for s in symbols ]
   else:
      styles = [ (c,s,l) for c in colors for s in symbols for l in ltypes ]
   return styles

# Plotting functions

def plotresults(filename, type=2, xscale='linear', showiter=False,
   showtol=True, style=None, limit=False, correct=[], latest=True, label=None):
   '''Returns:
   if plot requested: handle
   otherwise: (par,results,tolret,passes,cputime)

   filename = text file with simulation results
   type:  1 = BER
          2 = SER (as Hamming distance, default)
          3 = FER
          4 = GSER (as Levenshtein distance)
          5 = 3D profile (not supported)
          6 = 2D profile (as multiple graphs)
          7 = Burst-error profile
          8 = passes
          9 = cputime
          10 = cputime/pass
          or a string to match with header
   xscale:    'linear' (default) or 'log'
   showiter:  plot all iterations / indices in set? (default: false = show last)
              if true, show all indices
              if a list, interpret as indices to show
              NOTE: for profile, indices select positions to show results for
   showtol:   (T1-4,6) plot tolerance limits? (default: true)
   style:     if absent, do not plot results
              if a string, use this as the line format string
              if a list, use respectively as color, marker, linestyle
   limit:     limit number of markers? (default: false)
              if true, limits to 20-40 markers per line
              if a number 'n', one marker is printed for every 'n' points
   correct:   data set for BSID parameter correction (default: no correction)
              contains: [Ps, Pd, Pi, Reff]
   latest:    if true (default) load only the latest simulation in file
              if false, concatenate all simulations in file
              if a list, concatenate the specified simulations in file
   label:     legend label

   Note: types 1 & 2 use the same data set (symbol-error), but just
         change the axis labels this facilitates use with binary codes.
   '''

   # Load data from file, and reorganize as necessary
   (par,results,tolerance,passes,cputime,header,comments) = loadresults(filename,latest)
   # extract last system and date from comments
   system = getafter('Communication System:',comments)
   date = getafter('Date:',comments)
   # apply correction to BSID parameter if requested
   if correct != []:
      [Ps, Pd, Pi, Reff] = correct
      par = correct_bsid_to_normerror(par, Ps, Pd, Pi, Reff)
   # extract results to be plotted
   if type==1 or type==2 or type==3 or type==4:
      # extract columns according to type requested
      tag = { 1: 'BER', 2: 'SER', 3: 'FER', 4:'LD' }
      #i = [s.startswith(tag[type]) for s in header]
      (results,dd,hh) = get_columns(results,header,tag[type])
      (tolerance,dd,hh) = get_columns(tolerance,header,tag[type])
   elif type==7:
      # compute only P(e_i | e_{i-1}) / P(e_i)
      results = (results[:,2] / results[:,3]) / (results[:,1] + results[:,2])
   elif not (type >= 1 and type <= 10):
      # extract columns according to type requested
      (results,dd,hh) = get_columns(results,header,type)
      (tolerance,dd,hh) = get_columns(tolerance,header,type)
   ns=1
   # determine decimation factor to limit number of markers
   if limit == True: # limit to 20-40/line
      if type==6:
         pts = np.size(results,1)
      else:
         ptr = np.size(par,0)
      ns = max(1, pts/20)
   elif limit != False: # limit as requested by user
      ns = limit
      limit = True
   # keep results from requested iterations
   if isinstance(showiter, list):
      results = results[:,showiter]
      tolerance = tolerance[:,showiter]
   elif not showiter:
      # keep last result in set only
      results = results[:,-1]
      tolerance = tolerance[:,-1]

   # keep tolerance values to return
   tolret = tolerance
   # clear tolerances, if requested
   if not showtol:
      tolerance = []

   # Skip the figure plot if requested
   if style is None:
      return (par,results,tolret,passes,cputime)

   # do the appropriate plot
   if type==1 or type==2 or type==3 or type==4:
      ylabel = { 1: 'Bit Error Rate', 2: 'Symbol Error Rate', 3: 'Frame Error Rate', 4: 'Generalized Symbol Error Rate' }
      h = plotitem(par,results,tolerance,style,xscale,'log',ns,label)
      plt.xlabel('Channel Parameter')
      plt.ylabel(ylabel[type])
   elif type==5:
      print "Not supported"
      sys.exit(1)
      #plotprofile(par,0:(size(results,2)-1),results,tolerance,xscale,'linear','log')
      plt.xlabel('Channel Parameter')
      plt.ylabel('Value/Position')
      plt.zlabel('Symbol Error Rate')
   elif type==6:
      cols = np.size(results,1)
      # convert to numpy arrays and transpose
      results = np.array(results).transpose()
      tolerance = np.array(tolerance).transpose()
      label = '%s, p=%s' % (label, ','.join([ '%g' % n for n in par ]))
      h = plotitem(range(cols),results,tolerance,style,'linear','log',ns,label)
      plt.xlabel('Value/Position')
      plt.ylabel('Symbol Error Rate')
   elif type==7:
      h = plotitem(par,results,[],style,xscale,'log',ns,label)
      plt.xlabel('Channel Parameter')
      plt.ylabel(r'Burstiness $Pr[e_i|e_{i-1}] / Pr[e_i]$')
   elif type==8:
      # Plot the number of samples
      h = plotitem(par,passes,[],style,xscale,'log',ns,label)
      plt.ylabel('Number of Samples (Frames)')
   elif type==9:
      # Plot the CPU time used
      h = plotitem(par,cputime/3600,[],style,xscale,'log',ns,label)
      plt.ylabel('Compute Time Used (Hours)')
   elif type==10:
      # Plot the CPU time used per sample
      h = plotitem(par,cputime/passes,[],style,xscale,'log',ns,label)
      plt.ylabel('Compute Time per Sample (Seconds)')
   else: # catch-all
      h = plotitem(par,results,tolerance,style,xscale,'log',ns,label)
      plt.xlabel('Channel Parameter')
      plt.ylabel('Value')

   return h

def plotitem(x,y,ytol,style,xscale,yscale,ns=1,label=None):
   # ensure x,y,ytol are numpy arrays
   x = np.array(x).squeeze()
   y = np.array(y).squeeze()
   ytol = np.array(ytol).squeeze()
   # work on existing plot
   plt.hold(True)
   # debug output
   #print 'x = ', x
   #print 'y = ', y
   #print 'ytol = ', ytol
   # get number of columns in y-data
   if y.ndim == 1 or y.ndim == 0: # 1D vector or scalar
      curves = 1
   else:
      curves = np.size(y,1)
   # check for validity of style parameter
   if not isinstance(style, str):
      assert isinstance(style, (list, tuple))
      assert len(style) == 3
   # do the basic plot
   if len(ytol) == 0:
      if isinstance(style, str):
         h = plt.plot(x,y, style, markevery=ns, label=label)
      else:
         h = plt.plot(x,y, color=style[0], marker=style[1], markeredgecolor=style[0], markerfacecolor='none', linestyle=style[2], markevery=ns, label=label)
   else:
      if isinstance(style, str):
         h = plt.errorbar(x,y,ytol, fmt=style, markevery=ns, label=label)
      else:
         h = plt.errorbar(x,y,ytol, color=style[0], marker=style[1], markeredgecolor=style[0], markerfacecolor='none', linestyle=style[2], markevery=ns, label=label)
   # set linewidths to use
   if curves == 1:
      width = [0.5]
   else:
      width = np.linspace(0.25,1,curves)
   # set linewidths
   for i in range(curves):
      plt.setp(h[i], linewidth=width[i])
   # set axis scales
   plt.gca().set_xscale(xscale)
   plt.gca().set_yscale(yscale)
   return h

# fit polynomial of the form y = a x^b
def fitpoly(x, y):
   lgx = np.log(x)
   lgy = np.log(y)
   (b, lga) = np.polyfit(lgx, lgy, 1)
   a = np.exp(lga)
   return (b, a)

# fit polynomial of the form y = a x^b, where b is known
def fitpoly_constrained(x, y, b):
   lgx = np.log(x)
   lgy = np.log(y)
   lga = lgy - b*lgx
   a = np.exp(lga)
   a = np.mean(a)
   return a

def analyze(x, y):
   # change data to 1D arrays
   x = x.reshape((-1))
   y = y.reshape((-1))
   # extract region of interest
   mask = x < np.median(x)
   x = x[mask]
   y = y[mask]
   # shortcut for size-zero vectors
   if x.size == 0:
      return (np.nan, np.nan)
   # fit the polynomial
   (b,a) = fitpoly(x,y)
   b = np.round(b)
   a = fitpoly_constrained(x,y,b)
   return (b, a)

def plot_and_analyze(filename, type, style, rate, corrected):
   # plot settings
   xscale = 'log'
   showiter = False
   showtol = False
   limit = False
   if not corrected:
      correct = []
   else:
      correct = [0,1,1,rate]
   # confirm that file exists
   if not os.path.exists(filename):
      #print filename, 'does not exist'
      return None
   # analyze if needed
   if type==2: # SER
      (par,results,tolret,passes,cputime) = \
         plotresults(filename,type,xscale,showiter,showtol,None,limit,correct)
      (b,a) = analyze(par,results)
      print "%f\t%f\t%s" % (b,a,filename.split('.')[-2])
   # do the plot
   h = plotresults(filename,type,xscale,showiter,showtol,style,limit,correct)
   return h


def getafter(containing, list):
   s = getlastline(containing, list)
   return s[len(containing):].strip()

def getlastline(containing, list):
   list.reverse()
   for line in list:
      if line.startswith(containing):
         return line
   return ''

# TeX-formats the float value, returning string representation
def format_float(val, base=10):
   exp = int(np.floor(np.log(abs(val))/np.log(base)))
   mul = int(np.floor(val / pow(base,exp)))
   # major value
   if mul == 1:
      return r'%d^{%d}' % (base, exp)
   # minor value
   return r'%d\times%d^{%d}' % (mul, base, exp)

class tickformatter(tck.LogFormatter):
   def __call__(self, val, pos=None):
      if self._base == 10:
         exp = int(math.floor(math.log10(abs(val))))
         mul = int(math.floor(val / pow(10, exp)))
      else:
         exp = int(math.floor(math.log(abs(val), self._base)))
         mul = int(math.floor(val / pow(self._base, exp)))
      #print 'Formatter: %g = %dx%d^%d' % (val, mul, self._base, exp)
      # always show major labels
      if mul == 1:
         return r'$%d^{%d}$' % (self._base, exp)
      # skip minor labels unless they fall on axes limits
      lim = self.axis.get_view_interval()
      eps = np.finfo(float).eps
      if val > lim[0]+2*eps and val < lim[1]-2*eps:
         return ''
      return r'$%d\times%d^{%d}$' % (mul, self._base, exp)

def xlim(lo,hi,step=0):
   eps = np.finfo(float).eps
   # current axes only
   ha = plt.gca()
   ha.set_xlim(lo-eps,hi+eps)
   if step > 0:
      ha.set_xticks(np.arange(lo,hi+step,step))
   return

def ylim(lo,hi,step=0):
   eps = np.finfo(float).eps
   # current axes only
   ha = plt.gca()
   ha.set_ylim(lo-eps,hi+eps)
   if step > 0:
      ha.set_yticks(np.arange(lo,hi+step,step))
   return

def all_xlim(lo,hi,step=0):
   eps = np.finfo(float).eps
   # repeat for all axes
   for ha in plt.gcf().get_axes():
      ha.set_xlim(lo-eps,hi+eps)
      if step > 0:
         ha.set_xticks(np.arange(lo,hi+step,step))
   return

def all_ylim(lo,hi,step=0):
   eps = np.finfo(float).eps
   # repeat for all axes
   for ha in plt.gcf().get_axes():
      ha.set_ylim(lo-eps,hi+eps)
      if step > 0:
         ha.set_yticks(np.arange(lo,hi+step,step))
   return

def typeset(hf=[],figsize=(6.00,4.50),fontname='serif'):
   '''Returns: nothing

   hf = figure handle (defaults to current figure)
   figsize = (width,height) size tuple in inches (default 6 x 4.5)
   fontname = font to use (default to serif)
   '''

   if hf==[]:
      hf = plt.gcf()

   # set paper size
   hf.set_size_inches(figsize)
   # set fonts
   for ht in hf.texts:
      ht.set(fontname=fontname, fontsize=11)

   # repeat for all axes
   for ha in hf.get_axes():
      # set fonts
      ha.title.set(fontname=fontname, fontsize=11)
      ha.xaxis.label.set(fontname=fontname, fontsize=10)
      ha.yaxis.label.set(fontname=fontname, fontsize=10)
      # format any text lines
      for ht in ha.texts:
         ht.set(fontname=fontname, fontsize=9)
      # format any text lines in child artists (includes legends)
      for hl in ha.get_children():
         if hasattr(hl, 'texts'):
            for hlt in hl.texts:
               hlt.set(fontname=fontname, fontsize=9)
      # format ticks
      for ht in ha.xaxis.majorTicks + ha.yaxis.majorTicks:
         ht.label.set(fontname=fontname, fontsize=10)
      for ht in ha.xaxis.minorTicks + ha.yaxis.minorTicks:
         ht.label.set(fontname=fontname, fontsize=10)
      # update tick formatters on log plots
      if ha.xaxis.get_scale() == 'log':
         ha.xaxis.set_major_formatter(tickformatter())
         ha.xaxis.set_minor_formatter(tickformatter())
      if ha.yaxis.get_scale() == 'log':
         ha.yaxis.set_major_formatter(tickformatter())
         ha.yaxis.set_minor_formatter(tickformatter())

   # force redraw
   plt.draw()
   return

def shrink_axes(xfrac, yfrac=1.0, xoff=0.0, yoff=0.0):
   # shrink width of current axes by given fraction
   box = plt.gca().get_position()
   plt.gca().set_position([box.x0 + xoff, box.y0 + yoff, box.width * xfrac, box.height * yfrac])
   return

# function to load data from excel files

def loadexcel(fname, sheet, skiprows=0, skipcols=0):
   '''Returns: (data)

   Where:
      fname = file to be loaded
      sheet = name of sheet to load
      skiprows = number of rows to skip from top (=0)
      skipcols = number of cols to skip from left (=0)
      data = matrix of data values (as list of lists)
   '''

   book = xlrd.open_workbook(fname)
   data = []
   sh = book.sheet_by_name(sheet)
   for i in range(skiprows, sh.nrows):
      record = []
      for j in range(skipcols, sh.ncols):
         record.append(sh.cell(i,j).value)
      data.append(record)
   return data
