/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "dvbcrsc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;
using libbase::vector;

const libbase::serializer dvbcrsc::shelper("fsm", "dvbcrsc", dvbcrsc::create);

// TODO: may need to reverse the bits here
const int dvbcrsc::csct[7][8] = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 6, 4, 2, 7, 1,
      3, 5}, {0, 3, 7, 4, 5, 6, 2, 1}, {0, 5, 3, 6, 2, 7, 1, 4}, {0, 4, 1, 5,
      6, 2, 7, 3}, {0, 2, 5, 7, 1, 3, 4, 6}, {0, 7, 6, 1, 3, 4, 5, 2}};

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

dvbcrsc::dvbcrsc(const dvbcrsc& x)
   {
   // copy automatically what we can
   reg = x.reg;
   }

// finite state machine functions - resetting

void dvbcrsc::reset()
   {
   fsm::reset();
   reg = 0;
   }

void dvbcrsc::reset(vector<int> state)
   {
   fsm::reset(state);
   reg = fsm::convert(state, 2);
   }

void dvbcrsc::resetcircular(vector<int> zerostate, int n)
   {
   // TODO: check input state is valid
   // circulation state is obtainable only if the sequence length is not
   // a multiple of the period
   assert(n%7 != 0);
   // lookup the circulation state and set accordingly
   reset(fsm::convert(csct[n % 7][zerostate], nu, 2));
   }

// finite state machine functions - state advance etc.

void dvbcrsc::advance(vector<int>& input)
   {
   fsm::advance(input);
   // ref: ETSI EN 301 790 V1.4.1 (2005-04)
   // ip[0] = A, ip[1] = B
   assert(input(0) != fsm::tail && input(1) != fsm::tail);
   // process input
   bitfield ip(input);
   // compute the shift-register left input
   bitfield lsi = (reg + (ip(0) ^ ip(1))) * bitfield("1011");
   // do the shift
   reg = reg << lsi;
   // apply the second input
   reg ^= (ip(1) + ip(1) + bitfield("0"));
   }

vector<int> dvbcrsc::output(vector<int> input) const
   {
   // ref: ETSI EN 301 790 V1.4.1 (2005-04)
   // ip[0] = A, ip[1] = B
   assert(input(0) != fsm::tail && input(1) != fsm::tail);
   // process input
   bitfield ip(input);
   // compute the shift-register left input
   bitfield lsi = (reg + (ip(0) ^ ip(1))) * bitfield("1011");
   // determine output
   // since the code is systematic, the first (low-order) op is the input
   bitfield op = ip;
   // low-order parity is Y
   op = (reg + lsi) * bitfield("1101") + op;
   // next is W
   op = (reg + lsi) * bitfield("1001") + op;
   return op;
   }

vector<int> dvbcrsc::state() const
   {
   return vector<bool>(reg);
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

} // end namespace
