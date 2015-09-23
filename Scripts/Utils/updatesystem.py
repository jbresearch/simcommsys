#!/usr/bin/python

import sys
import re

## utility methods

def loaddata(fid):
   # read all lines in the file
   data = []
   for line in fid:
      line = line.rstrip()
      data.append(line)
   return data

def savedata(fid, data):
   # write all lines to the file
   for line in data:
      fid.write(line + '\n')
   return

## component updates

# convert bsc to qsc
def update_bsc(data):
   component = 'bsc'
   # confirm there is the required component
   if component not in data:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # write converted component
   result.append("qsc<bool>")
   # copy input after component
   result += data
   # return result
   return result

# convert bsid v.5 to qids v.2
def update_bsid(data):
   component = 'bsid'
   # confirm there is the required component
   if component not in data:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # read component
   assert data.pop(0) == "# Version"
   assert data.pop(0) == "5"
   assert data.pop(0) == "# Biased?"
   biased = data.pop(0)
   assert data.pop(0) == "# Vary Ps?"
   varyPs = data.pop(0)
   assert data.pop(0) == "# Vary Pd?"
   varyPd = data.pop(0)
   assert data.pop(0) == "# Vary Pi?"
   varyPi = data.pop(0)
   assert data.pop(0) == "# Cap on I (0=uncapped)"
   Icap = data.pop(0)
   assert data.pop(0) == "# Fixed Ps value"
   fixedPs = data.pop(0)
   assert data.pop(0) == "# Fixed Pd value"
   fixedPd = data.pop(0)
   assert data.pop(0) == "# Fixed Pi value"
   fixedPi = data.pop(0)
   assert data.pop(0) == "# Mode for receiver (0=trellis, 1=lattice)"
   lattice = data.pop(0)
   # write converted component
   result.append("qids<bool,float>")
   result.append("# Version")
   result.append("2")
   result.append("# Vary Ps?")
   result.append(varyPs)
   result.append("# Vary Pd?")
   result.append(varyPd)
   result.append("# Vary Pi?")
   result.append(varyPi)
   result.append("# Cap on I (0=uncapped)")
   result.append(Icap)
   result.append("# Fixed Ps value")
   result.append(fixedPs)
   result.append("# Fixed Pd value")
   result.append(fixedPd)
   result.append("# Fixed Pi value")
   result.append(fixedPi)
   result.append("# Mode for receiver (0=trellis, 1=lattice)")
   result.append(lattice)
   # copy input after component
   result += data
   # return result
   return result

# convert dminner2 v.4 to tvb v.4
def update_dminner2(data):
   component = 'dminner2'
   # find the lines with the required component
   matches = [line for line in data if line.startswith(component)]
   # confirm there is the required component
   if not matches:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   component = matches[0]
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # determine component template parameters
   regex = re.compile(r'\w+<(\w+)>')
   tparam = regex.findall(component)
   assert len(tparam) == 1
   # read component
   assert data.pop(0) == "# Version"
   assert data.pop(0) == "3"
   assert data.pop(0) == "# User threshold?"
   user_threshold = data.pop(0)
   if user_threshold == "1":
      assert data.pop(0) == "#: Inner threshold"
      th_inner = data.pop(0)
      assert data.pop(0) == "#: Outer threshold"
      th_outer = data.pop(0)
   else:
      th_inner = "0"
      th_outer = "0"
   assert data.pop(0) == "# Normalize metrics between time-steps?"
   norm = data.pop(0)
   assert data.pop(0) == "# n"
   n = data.pop(0)
   assert data.pop(0) == "# k"
   k = data.pop(0)
   assert data.pop(0) == "# codebook type (0=sparse, 1=user, 2=tvb)"
   codebook_type = data.pop(0)
   codebooks = []
   if codebook_type == "1":
      assert data.pop(0) == "#: codebook name"
      codebookname = data.pop(0)
      assert data.pop(0) == "#: codebook entries"
      codebook = []
      for d in range(2 ** int(k)):
         codebook.append(data.pop(0))
      codebooks.append(codebook)
   elif codebook_type == "2":
      assert data.pop(0) == "#: codebook name"
      codebookname = data.pop(0)
      assert data.pop(0) == "#: codebook count"
      num_codebooks = data.pop(0)
      for i in range(int(num_codebooks)):
         assert data.pop(0) == "#: codebook entries (table %d)" % i
         codebook = []
         for d in range(2 ** int(k)):
            codebook.append(data.pop(0))
         codebooks.append(codebook)
   assert data.pop(0) == "# marker type (0=random, 1=zero, 2=symbol-alternating, 3=mod-vectors)"
   marker_type = data.pop(0)
   marker_vectors = []
   if marker_type == "3":
      assert data.pop(0) == "#: modification vectors"
      num_marker_vectors = data.pop(0)
      for i in range(int(num_marker_vectors)):
         marker_vectors.append(data.pop(0))
   assert data.pop(0) == "# Version"
   assert data.pop(0) == "4"
   assert data.pop(0) == "# Use batch receiver computation?"
   batch = data.pop(0)
   assert data.pop(0) == "# Lazy computation of gamma?"
   lazy = data.pop(0)
   assert data.pop(0) == "# Global storage / caching of computed gamma values?"
   globalstore = data.pop(0)
   assert data.pop(0) == "# Number of codewords to look ahead when stream decoding"
   lookahead = data.pop(0)
   # write converted component
   result.append("tvb<bool,%s,float>" % tparam[0])
   result.append("# Version")
   result.append("4")
   result.append("#: Inner threshold")
   result.append(th_inner)
   result.append("#: Outer threshold")
   result.append(th_outer)
   result.append("# Normalize metrics between time-steps?")
   result.append(norm)
   result.append("# Use batch receiver computation?")
   result.append(batch)
   result.append("# Lazy computation of gamma?")
   result.append(lazy)
   result.append("# Global storage / caching of computed gamma values?")
   result.append(globalstore)
   result.append("# Number of codewords to look ahead when stream decoding")
   result.append(lookahead)
   result.append("# n")
   result.append(n)
   result.append("# k")
   result.append(k)
   result.append("# codebook type (0=sparse, 1=random, 2=user[seq], 3=user[ran])")
   if codebook_type == "0": # sparse
      result.append("0")
   else: # must be user or tvb
      result.append("2") # these were in sequential mode
      result.append("#: codebook name")
      result.append(codebookname)
      result.append("#: codebook count")
      result.append(str(len(codebooks)))
      for codebook in codebooks:
         result.append("#: codebook entries")
         for codeword in codebook:
            result.append(" ".join(codeword[::-1])) # output in reverse, with spaces between bits
   result.append("# marker type (0=zero, 1=random, 2=user[seq], 3=user[ran])")
   # old marker type (0=random, 1=zero, 2=symbol-alternating, 3=mod-vectors)"
   if marker_type == "0": # random
      result.append("1")
   elif marker_type == "1": # zero
      result.append("0")
   elif marker_type == "2": # symbol-alternating
      result.append("2")
      result.append("#: marker vectors")
      result.append("2")
      result.append("0" * int(n))
      result.append("1" * int(n))
   else: # must be mod-vectors
      result.append("2")
      result.append("#: marker vectors")
      result.append(str(len(marker_vectors)))
      for marker in marker_vectors:
         result.append(" ".join(marker[::-1])) # output in reverse, with spaces between bits
   # copy input after component
   result += data
   # return result
   return result

# upgrade serialization string for tvb (add real2 param)
def update_tvb(data):
   component = 'tvb'
   # find the lines with the required component
   matches = [line for line in data if line.startswith(component)]
   # confirm there is the required component
   if not matches:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   component = matches[0]
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # determine component template parameters (match 2 only)
   regex = re.compile(r'\w+<(\w+),(\w+)>')
   tparam = regex.findall(component)
   # add parameter if necessary and rewrite
   if tparam:
      assert len(tparam) == 1
      result.append("tvb<%s,%s,float>" % tparam[0])
   else:
      result.append(component)
   # copy input after component
   result += data
   # return result
   return result

# upgrade serialization string for qids (add real param)
def update_qids(data):
   component = 'qids'
   # find the lines with the required component
   matches = [line for line in data if line.startswith(component)]
   # confirm there is the required component
   if not matches:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   component = matches[0]
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # determine component template parameters (match 1 only)
   regex = re.compile(r'\w+<(\w+)>')
   tparam = regex.findall(component)
   # add parameter if necessary and rewrite
   if tparam:
      assert len(tparam) == 1
      result.append("qids<%s,float>" % tparam[0])
   else:
      result.append(component)
   # copy input after component
   result += data
   # return result
   return result

# upgrade serialization string for map_dividing (add dbl2 param)
def update_map_dividing(data):
   component = 'map_dividing'
   # find the lines with the required component
   matches = [line for line in data if line.startswith(component)]
   # confirm there is the required component
   if not matches:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   component = matches[0]
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # determine component template parameters (match 2 only)
   regex = re.compile(r'\w+<(\w+),(\w+)>')
   tparam = regex.findall(component)
   # add parameter if necessary and rewrite
   if tparam:
      assert len(tparam) == 1
      result.append("map_dividing<%s,%s,%s>" % (tparam[0][0], tparam[0][1], tparam[0][1]))
   else:
      result.append(component)
   # copy input after component
   result += data
   # return result
   return result

# upgrade serialization string for map_aggregating (add dbl2 param)
def update_map_aggregating(data):
   component = 'map_aggregating'
   # find the lines with the required component
   matches = [line for line in data if line.startswith(component)]
   # confirm there is the required component
   if not matches:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   component = matches[0]
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # determine component template parameters (match 2 only)
   regex = re.compile(r'\w+<(\w+),(\w+)>')
   tparam = regex.findall(component)
   # add parameter if necessary and rewrite
   if tparam:
      assert len(tparam) == 1
      result.append("map_aggregating<%s,%s,%s>" % (tparam[0][0], tparam[0][1], tparam[0][1]))
   else:
      result.append(component)
   # copy input after component
   result += data
   # return result
   return result

# upgrade serialization string for uncoded (replace with memoryless or update
# format to new uncoded, as needed)
def update_uncoded(data):
   component = 'uncoded<'
   # find the lines with the required component
   matches = [line for line in data if line.startswith(component)]
   # confirm there is the required component
   if not matches:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   component = matches[0]
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # do nothing if this is a new-format class
   if data[0] == "# Version":
      result.append(component)
      result += data
      return result
   # determine component template parameters (match 1 only)
   regex = re.compile(r'\w+<(\w+)>')
   tparam = regex.findall(component)
   assert len(tparam) == 1
   dbl = tparam[0]
   # make a copy of remaining data
   data_copy = list(data)
   # read component
   assert data.pop(0) == "# Encoder"
   assert data.pop(0) == "cached_fsm"
   assert data.pop(0) == "#: Base Encoder"
   encoder = data.pop(0)
   # determine alphabet size (match 1 only)
   regex = re.compile(r'\D+(\d+)\D+')
   tparam = regex.findall(encoder)
   assert len(tparam) == 1
   q = tparam[0]
   # assume uncoded
   uncoded = True
   # read encoder parameters
   if encoder.startswith("zsm"):
      assert data.pop(0) == "#: Repetition count"
      R = int(data.pop(0))
      if R > 1:
         uncoded = False
   else:
      assert data.pop(0) == "#: Generator matrix (k x n vectors)"
      k_by_n = data.pop(0)
      vlen = data.pop(0)
      vdat = data.pop(0)
      if k_by_n == "1\t1" and vlen == "1" and int(vdat) == 1:
         # this is uncoded; read the next (blank) line
         assert data.pop(0) == ""
      else:
         uncoded = False
   # read block length
   assert data.pop(0) == "# Block length"
   N = data.pop(0)
   # convert to memoryless if this is not an uncoded transmission
   if not uncoded:
      result.append("memoryless<%s>" % (dbl))
      result += data_copy
      return result
   # write converted component
   result.append("uncoded<%s>" % (dbl))
   result.append("# Version")
   result.append("1")
   result.append("# Alphabet size")
   result.append(q)
   result.append("# Block length")
   result.append(N)
   # copy input after component
   result += data
   # return result
   return result

# upgrade serialization string for commsys_stream (add real param)
def update_commsys_stream(data):
   component = 'commsys_stream'
   # find the lines with the required component
   matches = [line for line in data if line.startswith(component)]
   # confirm there is the required component
   if not matches:
      print "No %s in file" % component
      return data
   # copy input before component, and remove
   component = matches[0]
   i = data.index(component)
   result = data[0:i]
   data = data[i:]
   assert data.pop(0) == component
   # determine component template parameters (match 2 only)
   regex = re.compile(r'\w+<(\w+),(\w+)>')
   tparam = regex.findall(component)
   # add parameter if necessary and rewrite
   if tparam:
      assert len(tparam) == 1
      stype = tparam[0][0]
      container = tparam[0][1]
      # determine real type from channel
      matches = [line for line in data if line.startswith('qids')]
      assert matches
      channel = matches[0]
      regex = re.compile(r'\w+<(\w+),(\w+)>')
      tparam = regex.findall(channel)
      assert tparam
      assert tparam[0][0] == stype
      real = tparam[0][1]
      # write updated system line
      result.append("commsys_stream<%s,%s,%s>" % (stype, container, real))
   else:
      result.append(component)
   # copy input after component
   result += data
   # return result
   return result

## file processing

def process(file):
   print "Processing", file
   # input
   lines = loaddata(open(file, 'r'))
   # process
   lines = update_bsc(lines)
   lines = update_bsid(lines)
   lines = update_dminner2(lines)
   lines = update_tvb(lines)
   lines = update_qids(lines)
   lines = update_map_dividing(lines)
   lines = update_map_aggregating(lines)
   lines = update_uncoded(lines)
   lines = update_commsys_stream(lines)
   # output
   savedata(open(file, 'w'), lines)
   return

# main method

def main():
   files = sys.argv[1:]
   for file in files:
      process(file)

# Run main, if called from the command line
if __name__ == '__main__':
   main()
