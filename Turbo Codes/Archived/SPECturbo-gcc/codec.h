#ifndef __codec_h
#define __codec_h

#include "config.h"
#include "vcs.h"
#include "sigspace.h"
#include "matrix.h"
#include "vector.h"

extern const vcs codec_version;

/*
  Version 1.01 (5 Mar 1999)
  renamed the mem_order() function to tail_length(), which is more indicative of the true
  use of this function.

  Version 1.10 (7 Jun 1999)
  modulated sequence changed from matrix to vector, in order to simplify the implementation
  of puncturing. Now, num_symbols returns the length of the signal space vector.
*/
class codec {
public:
   virtual void seed(const int s) {};

   virtual void encode(vector<int>& source, vector<int>& encoded) = 0;
   virtual void modulate(vector<int>& encoded, vector<sigspace>& tx) = 0;
   virtual void transmit(vector<sigspace>& tx, vector<sigspace>& rx) = 0;

   virtual void demodulate(vector<sigspace>& rx) = 0;
   virtual void decode(vector<int>& decoded) = 0;

   virtual int block_size() = 0;
   virtual int num_inputs() = 0;
   virtual int num_symbols() = 0;
   virtual int tail_length() = 0;

   virtual int num_iter() { return 1; };
};

#endif

