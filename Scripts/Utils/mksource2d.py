#!/usr/bin/python

import sys
import random

# 2D random source creation (decimal format)

def mksource2d(q, cols, rows, blocks, fd=sys.stdout):
   for k in xrange(blocks):
      fd.write("# block %d\n" % k)
      for i in xrange(rows):
         for j in xrange(cols):
            fd.write("%d\t" % random.randint(0,q-1))
         fd.write("\n")
      fd.write("\n")
   return

def main():
   if len(sys.argv) < 4:
      print "Usage:", sys.argv[0], "<q> <cols> <rows> [<blocks=1> [<seed=0>]]"
      sys.exit()

   if len(sys.argv) >= 6:
      seed = int(sys.argv[5])
   else:
      seed = 0

   if len(sys.argv) >= 5:
      blocks = int(sys.argv[4])
   else:
      blocks = 1

   rows = int(sys.argv[3])
   cols = int(sys.argv[2])
   q = int(sys.argv[1])

   random.seed(seed)
   print "# Seed = %d" % seed
   mksource2d(q, cols, rows, blocks)

# Run main, if called from the command line
if __name__ == '__main__':
   main()
