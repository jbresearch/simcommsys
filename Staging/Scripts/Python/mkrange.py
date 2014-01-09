#!/usr/bin/python

import sys

def main():
   if len(sys.argv) < 4:
      print "Usage: ", sys.argv[0], " <start> <multiplier> <stop>"
      sys.exit()

   start = float(sys.argv[1])
   multiplier = float(sys.argv[2])
   stop = float(sys.argv[3])

   while start > stop:
      print start
      start = start * multiplier

# Run main, if called from the command line
if __name__ == '__main__':
   main()
