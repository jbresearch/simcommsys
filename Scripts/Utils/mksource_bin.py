#!/usr/bin/python

import sys
import random

# random source creation (binary format)

def dec2bin(x,n):
   """converts value 'x' to a binary string of length 'n'.
   """
   f = "{:0%db}" % n
   return f.format(x)

def mksource_bin(q, length, blocks, fd=sys.stdout):
   bits = int(math.log(q,2))
   assert(q == (1<<bits))
   for k in xrange(blocks):
      fd.write("# block %d\n" % k)
      for i in xrange(length):
         fd.write("%s\n" % dec2bin(random.randint(0,q-1), bits))
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
   mksource_bin(q, length, blocks)

# Run main, if called from the command line
if __name__ == '__main__':
   main()
