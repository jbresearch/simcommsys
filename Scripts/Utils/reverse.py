#!/usr/bin/python

import sys

def loaddata(fid):
   # read all lines in the file
   data = []
   for line in fid:
      line = line.rstrip()
      data.append(line)
   return data

def process(data):
   for i in range(len(data)):
      # reverse non comment line
      if len(data[i])>0:
         if data[i][0] != '#':
            data[i] = data[i][::-1]
   return data

def savedata(fid,data):
   # write all lines to the file
   for line in data:
      fid.write(line + '\n')
   return

def main():
   # input
   fin = sys.stdin
   if len(sys.argv) >= 2:
      fin = open(sys.argv[1], 'r')
   lines = loaddata(fin)
   fin.close()
   # process
   lines = process(lines)
   # output
   fout = sys.stdout
   if len(sys.argv) >= 3:
      fout = open(sys.argv[2], 'w')
   elif len(sys.argv) >= 2:
      fout = open(sys.argv[1], 'w')
   savedata(fout,lines)
   fout.close()
   return

# Run main, if called from the command line
if __name__ == '__main__':
   main()
