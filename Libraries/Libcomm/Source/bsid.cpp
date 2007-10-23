#include "bsid.h"
#include "secant.h"

namespace libcomm {

const libbase::vcs bsid::version("Binary Substitution, Insertion, and Deletion Channel module (bsid)", 1.20);

const libbase::serializer bsid::shelper("channel", "bsid", bsid::create);


// constructors / destructors

bsid::bsid(const double Pd, const double Pi)
   {
   assert(Pd >=0 && Pd <= 1);
   assert(Pi >=0 && Pi <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid::Pd = Pd;
   bsid::Pi = Pi;
   }

// channel parameter updates
void bsid::set_ps(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   libbase::secant Qinv(libbase::Q);
   const double x = Qinv(Ps);
   const double No = 1/(get_eb()*x*x);
   set_no(No);
   }
   
void bsid::set_pd(const double Pd)
   {
   assert(Pd >=0 && Pd <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid::Pd = Pd;
   }
   
void bsid::set_pi(const double Pi)
   {
   assert(Pi >=0 && Pi <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid::Pi = Pi;
   }

// handle functions

void bsid::compute_parameters(const double Eb, const double No)
   {
   // computes substitution probability assuming Eb/No describes an AWGN channel with hard-decision demodulation
   Ps = libbase::Q(1/sqrt(Eb*No));
   libbase::trace << "DEBUG (bsid): Ps = " << Ps << "\n";
   // probabilities of insertion and deletion have to be specified at creation time
   // or else using the class-specific update functions
   }
   
// channel handle functions

sigspace bsid::corrupt(const sigspace& s)
   {
   const double p = r.fval();
   //libbase::trace << "DEBUG (bsid): p(s) = " << p << "\n";
   if(p < Ps)
      return -s;
   return s;
   }

double bsid::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   if(tx != rx)
      return Ps;
   return 1-Ps;
   }

// channel functions

void bsid::transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx)
   {
   // We have initially no idea how long the received sequence will be, so we first determine the state sequence at every timestep
   // keeping track of (a) the number of insertions *before* given position, and (b) whether the given position is transmitted or deleted
   const int tau = tx.size();
   libbase::vector<int> insertions(tau);
   insertions = 0;
   libbase::vector<int> transmit(tau);
   transmit = 1;
   // determine state sequence
   for(int i=0; i<tau; i++)
      {
      double p;
      while((p = r.fval()) < Pi)
         insertions(i)++;
      if(p < (Pi+Pd))
         transmit(i) = 0;
      }
   // Initialize results vector
#ifndef NDEBUG
   if(tau < 10)
      {
      libbase::trace << "DEBUG (bsid): transmit = " << transmit << "\n";
      libbase::trace << "DEBUG (bsid): insertions = " << insertions << "\n";
      }
#endif
   rx.init(transmit.sum() + insertions.sum());
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0, j=0; i<tau; i++)
      {
      while(insertions(i)--)
         rx(j++) = (r.fval() < 0.5) ? sigspace(1,0) : sigspace(-1,0);
      if(transmit(i))
         rx(j++) = corrupt(tx(i));
      }
   }

void bsid::receive(const libbase::matrix<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const
   {
   // Compute sizes
   const int M = tx.ysize();
   const int m = rx.size()-1;
   // This implementation only works one transmission step at a time
   assert(tx.xsize() == 1);
   // Initialize results vector
   ptable.init(1, M);
   // Work out the probabilities of each possible signal
   for(int x=0; x<M; x++)
      ptable(0,x) = pdf(tx(0,x), rx(m));
   ptable *= (1-Pi-Pd);
   ptable += 0.5*Pi*Pd;
   ptable /= (1<<m)*(1-Pi)*(1-Pd);
   }

// description output

std::string bsid::description() const
   {
   return "BSID channel";
   }

// object serialization - saving

std::ostream& bsid::serialize(std::ostream& sout) const
   {
   sout << Pd << "\n";
   sout << Pi << "\n";
   return sout;
   }

// object serialization - loading

std::istream& bsid::serialize(std::istream& sin)
   {
   sin >> Pd;
   sin >> Pi;
   return sin;
   }

}; // end namespace
