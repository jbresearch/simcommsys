#include "dvbcrsc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

const libbase::vcs dvbcrsc::version("DVB-Standard Circular Recursive Systematic Convolutional Coder module (dvbcrsc)", 1.20);

const libbase::serializer dvbcrsc::shelper("fsm", "dvbcrsc", dvbcrsc::create);

const int dvbcrsc::csct[7][8] = {{0,1,2,3,4,5,6,7}, {0,6,4,2,7,1,3,5}, {0,3,7,4,5,6,2,1}, \
   {0,5,3,6,2,7,1,4}, {0,4,1,5,6,2,7,3}, {0,2,5,7,1,3,4,6}, {0,7,6,1,3,4,5,2}};

const int dvbcrsc::k = 2;
const int dvbcrsc::n = 4;
const int dvbcrsc::nu = 3;

// initialization

void dvbcrsc::init()
   {
   // create shift register
   reg.resize(nu);
   }

// constructors / destructors

dvbcrsc::dvbcrsc()
   {
   }

dvbcrsc::dvbcrsc(const dvbcrsc& x)
   {
   // copy automatically what we can
   reg = x.reg;
   }
   
dvbcrsc::~dvbcrsc()
   {
   }
   
// finite state machine functions - resetting

void dvbcrsc::reset(int state)
   {
   reg = state;
   N = 0;
   }

void dvbcrsc::resetcircular(int zerostate, int n)
   {
   assert(zerostate >= 0 && zerostate <= 7);
   // circulation state is obtainable only if the sequence length is not
   // a multiple of the period
   assert(n%7 != 0);
   // lookup the circulation state and set accordingly
   reset(csct[n%7][zerostate]);
   }

void dvbcrsc::resetcircular()
   {
   resetcircular(state(),N);
   }

// finite state machine functions - state advance etc.

void dvbcrsc::advance(int& input)
   {
   using libbase::bitfield;
   // ref: ETSI EN 301 790 V1.4.1 (2005-04)
   // ip[0] = A, ip[1] = B
   assert(input != fsm::tail);
   // process input
   bitfield ip;
   ip.resize(k);
   ip = input;
   // compute the shift-register left input
   bitfield lsi = ((ip[0]^ip[1]) + reg) * bitfield("1101");
   // do the shift
   reg = lsi >> reg;
   // apply the second input
   reg ^= (bitfield("0") + ip[1] + ip[1]);
   // increment the sequence counter
   N++;
   }

int dvbcrsc::output(const int& input) const
   {
   using libbase::bitfield;
   // ref: ETSI EN 301 790 V1.4.1 (2005-04)
   // ip[0] = A, ip[1] = B
   assert(input != fsm::tail);
   // process input
   bitfield ip;
   ip.resize(k);
   ip = input;
   // compute the shift-register left input
   bitfield lsi = ((ip[0]^ip[1]) + reg) * bitfield("1101");
   // determine output
   // since the code is systematic, the first (low-order) op is the input
   bitfield op = ip;
   // low-order parity is Y
   op = (lsi + reg) * bitfield("1011") + op;
   // next is W
   op = (lsi + reg) * bitfield("1001") + op;
   return op;
   }

int dvbcrsc::step(int& input)
   {
   int op = output(input);
   advance(input);
   return op;
   }

int dvbcrsc::state() const
   {
   return reg;
   }

// description output

std::string dvbcrsc::description() const
   {
   std::ostringstream sout;
   sout << "DVB-Standard Circular RSC Code";
   return sout.str();
   }

// object serialization - saving

std::ostream& dvbcrsc::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& dvbcrsc::serialize(std::istream& sin)
   {
   init();
   return sin;
   }

}; // end namespace
