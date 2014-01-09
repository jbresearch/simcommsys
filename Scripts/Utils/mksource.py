#!/usr/bin/python

import sys
import random

# random source creation (decimal format)

def mksource(q, length, blocks, fd=sys.stdout):
   for k in xrange(blocks):
      fd.write("# block %d\n" % k)
      for i in xrange(length):
         fd.write("%d\n" % random.randint(0,q-1))
   return

def main():
   if len(sys.argv) < 3:
      print "Usage:", sys.argv[0], "<q> <block-length> [<blocks=1> [<seed=0>]]"
      sys.exit()

   if len(sys.argv) >= 5:
      seed = int(sys.argv[4])
   else:
      seed = 0

   if len(sys.argv) >= 4:
      blocks = int(sys.argv[3])
   else:
      blocks = 1

   length = int(sys.argv[2])
   q = int(sys.argv[1])

   random.seed(seed)
   print "# Seed = %d" % seed
   mksource(q, length, blocks)

# Run main, if called from the command line
if __name__ == '__main__':
   main()
