/*!
 \file

 \par Version Control:
 - $Revision$
 - $Date$
 - $Author$
 */

#include "bsid2d.h"
#include "secant.h"
#include <sstream>
#include <limits>

namespace libcomm {

const libbase::serializer bsid2d::shelper("channel", "bsid2d", bsid2d::create);

// Internal functions

/*!
 \brief Initialization

 Sets the channel with \f$ P_s = P_d = P_i = 0 \f$. This way, any
 of the parameters not flagged to change with channel SNR will remain zero.
 */
void bsid2d::init()
   {
   // channel parameters
   Ps = 0;
   Pd = 0;
   Pi = 0;
   // set block size to unusable value
   N = 0;
   M = 0;
   //precompute();
   }

/*!
 \brief Determine state-machine values for a single timestep

 Returns the number of insertions \e before given position, and whether the
 given position is transmitted or deleted.
 */
void bsid2d::computestate(int& insertions, bool& transmit)
   {
   // initialize accumulators
   insertions = 0;
   transmit = true;
   // determine state sequence for this timestep
   double p;
   while ((p = r.fval()) < Pi)
      insertions++;
   if (p < (Pi + Pd))
      transmit = false;
   }

/*!
 \brief Determine state-machine values for a whole block

 \sa computestate();
 */
void bsid2d::computestate(array2i_t& insertions, array2b_t& transmit)
   {
   // determine matrix sizes
   const int M = insertions.size().rows();
   const int N = insertions.size().cols();
   assertalways(transmit.size().rows() == M);
   assertalways(transmit.size().cols() == N);
   // iterate over all timesteps
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         computestate(insertions(i, j), transmit(i, j));
   }

/*!
 \brief Accumulate matrix elements over given dimension
 \param m Matrix containing data to be accumulated
 \param dim Dimension over which to accumulate (0 or 1)
 */
void bsid2d::cumsum(array2i_t& m, int dim)
   {
   assert(dim==0 || dim==1);
   // determine matrix sizes
   const int M = m.size().rows();
   const int N = m.size().cols();
   // iterate over all matrix elements
   if (dim == 0)
      {
      for (int j = 0; j < N; j++)
         for (int i = 1; i < M; i++)
            m(i, j) += m(i - 1, j);
      }
   else
      {
      for (int i = 0; i < M; i++)
         for (int j = 1; j < N; j++)
            m(i, j) += m(i, j - 1);
      }
   }

// Constructors / Destructors

/*!
 \brief Principal constructor

 \sa init()
 */
bsid2d::bsid2d(const bool varyPs, const bool varyPd, const bool varyPi) :
   varyPs(varyPs), varyPd(varyPd), varyPi(varyPi)
   {
   // channel update flags
   assert(varyPs || varyPd || varyPi);
   // other initialization
   init();
   }

// Channel parameter handling

/*!
 \brief Set channel parameter

 This function sets any of Ps, Pd, or Pi that are flagged to change. Any of these
 parameters that are not flagged to change will instead be set to zero. This ensures
 that there is no leakage between successive uses of this class. (i.e. once this
 function is called, the class will be in a known determined state).
 */
void bsid2d::set_parameter(const double p)
   {
   set_ps(varyPs ? p : 0);
   set_pd(varyPd ? p : 0);
   set_pi(varyPi ? p : 0);
   libbase::trace << "DEBUG (bsid2d): Ps = " << Ps << ", Pd = " << Pd
         << ", Pi = " << Pi << "\n";
   }

/*!
 \brief Get channel parameter

 This returns the value of the first of Ps, Pd, or Pi that are flagged to change.
 If none of these are flagged to change, this constitutes an error condition.
 */
double bsid2d::get_parameter() const
   {
   if (varyPs)
      return Ps;
   if (varyPd)
      return Pd;
   if (varyPi)
      return Pi;
   std::cerr << "ERROR: BSID channel has no parameters\n";
   exit(1);
   }

// Channel parameter setters

void bsid2d::set_ps(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   bsid2d::Ps = Ps;
   }

void bsid2d::set_pd(const double Pd)
   {
   assert(Pd >=0 && Pd <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid2d::Pd = Pd;
   //precompute();
   }

void bsid2d::set_pi(const double Pi)
   {
   assert(Pi >=0 && Pi <= 1);
   assert(Pi+Pd >=0 && Pi+Pd <= 1);
   bsid2d::Pi = Pi;
   //precompute();
   }

void bsid2d::set_blocksize(int M, int N)
   {
   if (M != bsid2d::M || N != bsid2d::N)
      {
      assert(M > 0);
      bsid2d::M = M;
      assert(N > 0);
      bsid2d::N = N;
      //precompute();
      }
   }

// Channel function overrides

/*!
 \copydoc channel::corrupt()

 \note Due to limitations of the interface, which was designed for substitution channels,
 only the substitution part of the channel model is handled here.

 For the purposes of this channel, a \e substitution corresponds to a symbol inversion.
 This corresponds to the \f$ 0 \Leftrightarrow 1 \f$ binary substitution when used with BPSK
 modulation. For MPSK modulation, this causes the output to be the symbol farthest away
 from the input.
 */
bool bsid2d::corrupt(const bool& s)
   {
   const double p = r.fval();
   if (p < Ps)
      return !s;
   return s;
   }

// Channel functions

/*!
 \copydoc channel::transmit()

 The channel model implemented is described by independent state machines
 for the row and column sequences, each according to the following state
 diagram:
 \dot
 digraph bsid2dstates {
 // Make figure left-to-right
 rankdir = LR;
 // state definitions
 this [ shape=circle, color=gray, style=filled, label="t(i)" ];
 next [ shape=circle, color=gray, style=filled, label="t(i+1)" ];
 // path definitions
 this -> Insert [ label="Pi" ];
 Insert -> this;
 this -> Delete [ label="Pd" ];
 Delete -> next;
 this -> Transmit [ label="1-Pi-Pd" ];
 Transmit -> next [ label="1-Ps" ];
 Transmit -> Substitute [ label="Ps" ];
 Substitute -> next;
 }
 \enddot

 The cumulative number of insertions-deletions+transmissions for each row
 and column determine the final horizontal and vertical position respectively
 for the given bit. The bit is only actually transmitted if the transmit
 flags for the horizontal and vertical state machines are both true. The
 final matrix is padded to the smallest rectangle that will fit all final
 positions. Inserted bits and other padding are assumed to be equiprobable.

 \note We have initially no idea how long the received sequence will be, so
 we first determine the state sequence at every timestep, keeping
 track of:
 - the number of insertions \e before given position, and
 - whether the given position is transmitted or deleted.

 \note We have to make sure that we don't corrupt the array we're reading
 from (in the case where tx and rx are the same array); therefore,
 the result is first created as a new array and only copied over at
 the end.

 \sa corrupt()
 */
void bsid2d::transmit(const array2b_t& tx, array2b_t& rx)
   {
   // determine matrix sizes
   const int M = tx.size().rows();
   const int N = tx.size().cols();
   // compute state tables
   array2i_t insertions_h(M, N), insertions_v(M, N);
   array2b_t transmit_h(M, N), transmit_v(M, N);
   computestate(insertions_h, transmit_h);
   computestate(insertions_v, transmit_v);
   // initialize coordinate transformations
   array2i_t ii(M, N), jj(M, N);
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         {
         ii(i, j) = insertions_v(i, j) + (transmit_v(i, j) ? 0 : -1);
         jj(i, j) = insertions_h(i, j) + (transmit_h(i, j) ? 0 : -1);
         }
   // accumulate differential
   cumsum(ii, 0);
   cumsum(jj, 1);
   // compute final coordinates
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         {
         ii(i, j) += i;
         jj(i, j) += j;
         }
   // invert coordinates for transmitted bits
   const int MM = ii.max() + 1;
   const int NN = jj.max() + 1;
   array2i_t iii(MM, NN), jjj(MM, NN);
   iii = -1;
   jjj = -1;
   for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
         {
         if (transmit_h(i, j) & transmit_v(i, j))
            {
            iii(ii(i, j), jj(i, j)) = i;
            jjj(ii(i, j), jj(i, j)) = j;
            }
         }
   // Initialize results vector
   array2b_t newrx(MM, NN);
   // Corrupt the modulation symbols (simulate the channel)
   for (int i = 0; i < MM; i++)
      for (int j = 0; j < NN; j++)
         {
         if (iii(i, j) == -1) // insertion or padding
            {
            assert(jjj(i,j) == -1);
            newrx(i, j) = (r.fval() < 0.5);
            }
         else // transmission
            {
            assert(jjj(i,j) >= 0);
            newrx(i, j) = corrupt(tx(iii(i, j), jjj(i, j)));
            }
         }
   // copy results back
   rx = newrx;
   }

// description output

std::string bsid2d::description() const
   {
   std::ostringstream sout;
   sout << "2D BSID channel (" << varyPs << varyPd << varyPi << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& bsid2d::serialize(std::ostream& sout) const
   {
   sout << 1 << "\n";
   sout << varyPs << "\n";
   sout << varyPd << "\n";
   sout << varyPi << "\n";
   return sout;
   }

// object serialization - loading

std::istream& bsid2d::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> version;
   // read parameters
   sin >> varyPs;
   sin >> varyPd;
   sin >> varyPi;
   init();
   return sin;
   }

} // end namespace
