#ifndef __bpsk_h
#define __bpsk_h
      
#include "config.h"
#include "vcs.h"
#include "modulator.h"

extern const vcs bpsk_version;

class bpsk : public modulator {
public:
   bpsk();
   ~bpsk();
};

#endif
