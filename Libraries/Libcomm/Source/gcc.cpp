#include "gcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

const libbase::vcs gcc::version("Generalized Convolutional Coder module (gcc)", 1.11);

const libbase::serializer gcc::shelper("fsm", "gcc", gcc::create);

using libbase::bitfield;
using libbase::vector;

// initialization

void gcc::init(const vector<bitfield>& A, const vector<bitfield>& B, const vector<bitfield>& C, const vector<bitfield>& D)
   {
   // copy automatically what we can
   gcc::A = A;
   gcc::B = B;
   gcc::C = C;
   gcc::D = D;
   // determine code parameters from vector sizes where possible
   nu = A.size();
   k = B(0).size();
   n = C.size();
   // create shift register
   reg.resize(nu);
   // confirm consistency in input
   int i;
   assert(A.size() == nu);
   for(i=0; i<nu; i++)
      assert(A(i).size() == nu);
   assert(B.size() == nu);
   for(i=0; i<nu; i++)
      assert(B(i).size() == k);
   assert(C.size() == n);
   for(i=0; i<nu; i++)
      assert(C(i).size() == nu);
   assert(D.size() == n);
   for(i=0; i<nu; i++)
      assert(D(i).size() == k);
   }

// constructors / destructors

gcc::gcc()
   {
   }

gcc::gcc(const vector<bitfield>& A, const vector<bitfield>& B, const vector<bitfield>& C, const vector<bitfield>& D)
   {
   init(A,B,C,D);
   }

gcc::gcc(const gcc& x)
   {
   // copy automatically what we can
   k = x.k;
   n = x.n;
   nu = x.nu;
   A = x.A;
   B = x.B;
   C = x.C;
   D = x.D;
   reg = x.reg;
   }
   
gcc::~gcc()
   {
   }
   
// helpers - matrix multiplication

vector<bitfield> gcc::multiply(const vector<bitfield>& A, const vector<bitfield>& Bt) const
   {
   vector<bitfield> C;
   C.init(A);
   for(int i=0; i<A.size(); i++)
      {
      C(i).resize(0);
      for(int j=0; i<Bt.size(); j++)
         C(i) = C(i) + A(i) * B(j);
      }
   return C;
   }

// finite state machine functions - resetting

void gcc::reset(int state)
   {
   reg = state;
   }

void gcc::resetcircular(int zerostate, int n)
   {
   int i,j;
   // first get the transpose of A
   vector<bitfield> At;
   At.init(nu);
   for(i=0; i<nu; i++)
      {
      At(i).resize(0);
      for(j=0; j<nu; j++)
         At(i) = At(i) + A(j).extract(nu-1-i);
      }
   // now compute A^n by repeated matrix multiplication
   vector<bitfield> An = A;
   for(i=1; i<n; i++)
      An = multiply(An, At);
   // then add unit matrix
   bitfield b;
   b.resize(nu);
   b >>= bitfield("1");
   for(i=0; i<nu; i++, b>>=1)
      An(i) |= b;
   }

void gcc::resetcircular()
   {
   assert("Function not implemented.");
   }

// finite state machine functions - state advance etc.

void gcc::advance(int& input)
   {
   // process input
   bitfield ip;
   ip.resize(k);
   ip = input;
   // determine next state
   bitfield newstate;
   newstate.resize(0);
   for(int i=0; i<nu; i++)
      newstate = newstate + (A(i) * reg ^ B(i) * ip);
   reg = newstate;
   }

int gcc::output(int& input)
   {
   // process input
   bitfield ip;
   ip.resize(k);
   ip = input;
   // determine output
   bitfield op;
   op.resize(0);
   for(int i=0; i<n; i++)
      op = op + (C(i) * reg ^ D(i) * ip);
   return op;
   }

int gcc::step(int& input)
   {
   int op = output(input);
   advance(input);
   return op;
   }

int gcc::state() const
   {
   return reg;
   }

// description output

std::string gcc::description() const
   {
   int i;
   std::ostringstream sout;
   sout << "Generalized Convolutional Code (K=" << nu+1 << ", rate " << k << "/" << n << ", A=[";
   for(i=0; i<nu; i++)
      sout << A(i) << (i==nu-1 ? "], B=[" : "; ");
   for(i=0; i<nu; i++)
      sout << B(i) << (i==nu-1 ? "], C=[" : "; ");
   for(i=0; i<n; i++)
      sout << C(i) << (i==n-1 ? "], D=[" : "; ");
   for(i=0; i<n; i++)
      sout << D(i) << (i==n-1 ? "])" : "; ");
   return sout.str();
   }

// object serialization - saving

std::ostream& gcc::serialize(std::ostream& sout) const
   {
   sout << A << B << C << D;
   return sout;
   }

// object serialization - loading

std::istream& gcc::serialize(std::istream& sin)
   {
   sin >> A >> B >> C >> D;
   init(A,B,C,D);
   return sin;
   }

}; // end namespace
