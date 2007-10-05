#include "bpsk.h"

const vcs bpsk_version("BPSK Modulator module (bpsk)", 1.00);

bpsk::bpsk()
   {
   M = 2;
   s = new sigspace[M];
   s[0] = sigspace(-1, 0);
   s[1] = sigspace(+1, 0);
   }

bpsk::~bpsk()
   {
   delete[] s;
   }
