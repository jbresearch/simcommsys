#!/usr/bin/python

import sys

# marker code creation (binary format)

def dec2bin(x,n):
   """converts value 'x' to a binary string of length 'n'.
   """
   f = "{:0%db}" % n
   return f.format(x)

def mkmarker(d, marker, fd=sys.stdout):
   for x in xrange(1<<d):
      fd.write("%s%s\n" % (dec2bin(x,d), marker))
   return

def main():
   if len(sys.argv) != 3:
      print "Usage:", sys.argv[0], "<d> <marker>"
      sys.exit()

   marker = sys.argv[2]
   d = int(sys.argv[1])

   mkmarker(d, marker)

# Run main, if called from the command line
if __name__ == '__main__':
   main()
