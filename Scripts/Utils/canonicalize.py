#!/usr/bin/python

import sys
import os
import shutil
import commands

def graceful_execute(cmd):
   st, out = commands.getstatusoutput(cmd)
   if st != 0:
      retval = os.WEXITSTATUS(st)
      print "Error executing command:"
      print cmd
      print "Status:", retval
      print "Output:"
      print out
      return None
   return out

def get_type_and_container(fname):
   # look for a system
   system = None
   for line in open(fname):
      if line.startswith('commsys'):
         system = line.strip()
         break
   # if we found a system
   if system:
      # extract commsys template parameters
      par = system.split('<',1)[1].rsplit('>',1)[0].split(',')
      type = par[0]
      if len(par) > 1:
         container = par[1]
      else:
         container = 'vector'
      # verify parameters are valid
      type_list = ['erasable<bool>', 'bool', 'gf2', 'gf4', 'gf8', 'gf16', 'gf32', 'gf64', 'gf128', 'gf256', 'gf512', 'gf1024', 'sigspace']
      container_list = ['vector', 'matrix']
      if type in type_list and  container in container_list:
         return type, container
   # fallback
   print "No valid system found, skipping file."
   return None

def process(executable,file):
   print "Processing", file
   tmpfile = '/dev/shm/canonical.txt'
   system = get_type_and_container(file)
   if system:
      type, container = system
      cmd = "%s -t '%s' -c '%s' < '%s' > '%s'" % (executable, type, container, file, tmpfile)
      output = graceful_execute(cmd)
      if output is not None:
         shutil.move(tmpfile,file)
         print output
         return True
      else:
         return False
   return None

def main():
   if len(sys.argv) < 2:
      print "Usage:", os.path.basename(sys.argv[0]), "<tag>", "<files>..."
      return
   # interpret use parameters
   tag = sys.argv[1]
   files = sys.argv[2:]
   # determine executable to use
   executable = "canonicalize.%s.release" % tag
   if not graceful_execute("which %s" % executable):
      print "Executable not found - bad tag? (%s)" % tag
      sys.exit(1)
   # main process
   print "Starting canonicalization process for %d files..." % len(files)
   table = []
   for file in files:
      if process(executable,file) == False:
         table.append(file)
   # show list of problem files
   if table:
      print "Problems encountered with %d files:" % len(table)
      for file in table:
         print file
   return

# Run main, if called from the command line
if __name__ == '__main__':
   main()
