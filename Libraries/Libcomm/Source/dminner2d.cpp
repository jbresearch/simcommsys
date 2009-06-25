/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "dminner2d.h"
#include "bsid2d.h"
#include "dminner2.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show pilot & received rows/cols as they are being decoded
// 3 - Also show input/output probability tables
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 3
#endif

/*!
   \copydoc blockmodem::advance()

   Updates 2D pilot sequence
*/

template <class real, bool norm>
void dminner2d<real,norm>::advance() const
   {
   // Inherit sizes
   const int M = this->input_block_size().y;
   const int N = this->input_block_size().x;
   // Initialize space
   pilot.init(N*n,M*m);
   // creates 'tau' elements of 'n' bits each
   for(int i=0; i<pilot.ysize(); i++)
      for(int j=0; j<pilot.xsize(); j++)
         pilot(j,i) = (r.ival(2) != 0);
   }

/*!
   \copydoc blockmodem::domodulate()

   Updates 2D pilot sequence
*/

template <class real, bool norm>
void dminner2d<real,norm>::domodulate(const int q, const libbase::matrix<int>& encoded, libbase::matrix<bool>& tx)
   {
   // Each 'encoded' symbol must be representable by a single sparse matrix
   assertalways(this->q == q);
   // Inherit sizes
   const int M = this->input_block_size().y;
   const int N = this->input_block_size().x;
   // Check validity
   assertalways(M == encoded.ysize());
   assertalways(N == encoded.xsize());
   // Initialise result vector
   tx = pilot;
   assert(tx.ysize() == M*m);
   assert(tx.xsize() == N*n);
   // Encode source
   for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
         for(int ii=0; ii<m; ii++)
            for(int jj=0; jj<n; jj++)
               tx(j*n+jj,i*m+ii) ^= lut(encoded(j,i))(jj,ii);
   }

/*!
   \copydoc blockmodem::dodemodulate()

   Wrapper for decoder function assuming equiprobable APP table.
*/

template <class real, bool norm>
void dminner2d<real,norm>::dodemodulate(const channel<bool,libbase::matrix>& chan, const libbase::matrix<bool>& rx, libbase::matrix<array1d_t>& ptable)
   {
   // Inherit sizes
   const int M = this->input_block_size().y;
   const int N = this->input_block_size().x;
   // Create equiprobable a-priori probability table
   libbase::matrix<array1d_t> app(N,M);
   for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
         {
         app(j,i).init(q);
         app(j,i) = 1.0;
         }
   // Now call the full decode function
   dodemodulate(chan, rx, app, ptable);
   }

/*!
   \copydoc blockmodem::dodemodulate()

   Decodes 2D sequence by performing iterative row/column decodings.
*/

template <class real, bool norm>
void dminner2d<real,norm>::dodemodulate(const channel<bool,libbase::matrix>& chan, const libbase::matrix<bool>& rx, const libbase::matrix<array1d_t>& app, libbase::matrix<array1d_t>& ptable)
   {
   // Inherit sizes
   const int M = this->input_block_size().y;
   const int N = this->input_block_size().x;
   // Check input validity
   assertalways(M == app.ysize());
   assertalways(N == app.xsize());
   // Copy channel and create a 1D one with same parameters
   bsid2d theirchan = dynamic_cast<const bsid2d&>(chan);
   bsid mychan;
   mychan.set_ps(theirchan.get_ps());
   mychan.set_pd(theirchan.get_pd());
   mychan.set_pi(theirchan.get_pi());
   // Initialize result vector
   ptable = app;
   // Temporary variables
   libbase::vector<bool> rxvec;
   libbase::vector<bool> wsvec;
   libbase::vector<array1d_t> pin;
   libbase::vector<array1d_t> pout;
   libbase::vector<array1d_t> pacc;
   dminner2<real,norm> rowdec(n,int(log2(q)));
   dminner2<real,norm> coldec(m,int(log2(q)));
   rowdec.set_thresholds(0,0);
   coldec.set_thresholds(0,0);
   // Iterate a few times
   for(int k=0; k<5; k++)
      {
      // Decode rows
      mychan.set_blocksize(N);
      for(int i=0; i<M; i++)
         {
         ptable.extractrow(pin,i);
         // initialize storage
         pacc.init(pin);
         for(int ii=0; ii<pacc.size(); ii++)
            pacc(ii).init(pin(ii));
         // initialize value
         pacc = 1;
         for(int ii=0; ii<m; ii++)
            {
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): Decoding row " << ii << " of symbol row " << i << "\n";
#endif
            // set up row decoder's sparse alphabet & pilot sequence
            rowdec.set_lut(get_alphabet_row(ii));
            pilot.extractrow(wsvec,i*m+ii);
            rowdec.set_pilot(wsvec);
            rx.extractrow(rxvec,i*m+ii);
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): pilot = " << wsvec;
            libbase::trace << "DEBUG (dminner2d): rx = " << rxvec;
#endif
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pin = " << pin;
#endif
            rowdec.demodulate(mychan, rxvec, pin, pout);
            pacc *= pout;
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pout = " << pout;
            libbase::trace << "DEBUG (dminner2d): pacc = " << pacc;
#endif
            }
         ptable.insertrow(pacc,i);
         }
      // Decode columns
      mychan.set_blocksize(M);
      for(int j=0; j<N; j++)
         {
         ptable.extractcol(pin,j);
         // initialize storage
         pacc.init(pin);
         for(int jj=0; jj<pacc.size(); jj++)
            pacc(jj).init(pin(jj));
         // initialize value
         pacc = 1;
         for(int jj=0; jj<n; jj++)
            {
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): Decoding col " << jj << " of symbol col " << j << "\n";
#endif
            // set up col decoder's sparse alphabet & pilot sequence
            coldec.set_lut(get_alphabet_col(jj));
            pilot.extractcol(wsvec,j*n+jj);
            coldec.set_pilot(wsvec);
            rx.extractcol(rxvec,j*n+jj);
#if DEBUG>=2
            libbase::trace << "DEBUG (dminner2d): pilot = " << wsvec;
            libbase::trace << "DEBUG (dminner2d): rx = " << rxvec;
#endif
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pin = " << pin;
#endif
            coldec.demodulate(mychan, rxvec, pin, pout);
            pacc *= pout;
#if DEBUG>=3
            libbase::trace << "DEBUG (dminner2d): pout = " << pout;
            libbase::trace << "DEBUG (dminner2d): pacc = " << pacc;
#endif
            }
         ptable.insertcol(pacc,j);
         }
      }
   }

/*!
   \brief Confirm that LUT is valid
   Checks that all LUT entries are of the correct size and that there are no
   duplicate entries.
*/

template <class real, bool norm>
void dminner2d<real,norm>::validatelut() const
   {
   assertalways(lut.size() == num_symbols());
   for(int i=0; i<lut.size(); i++)
      {
      // all entries should be of the correct size
      assertalways(lut(i).ysize() == m);
      assertalways(lut(i).xsize() == n);
      // all entries should be distinct
      for(int j=0; j<i; j++)
         assertalways(lut(i).isnotequalto(lut(j)));
      }
   }

template <class real, bool norm>
libbase::vector<libbase::bitfield> dminner2d<real,norm>::get_alphabet_row(int i) const
   {
   libbase::vector<libbase::bitfield> lutb(lut.size());
   for(int k=0; k<lut.size(); k++)
      lutb(k) = libbase::bitfield(lut(k).extractrow(i));
   return lutb;
   }

template <class real, bool norm>
libbase::vector<libbase::bitfield> dminner2d<real,norm>::get_alphabet_col(int j) const
   {
   libbase::vector<libbase::bitfield> lutb(lut.size());
   for(int k=0; k<lut.size(); k++)
      lutb(k) = libbase::bitfield(lut(k).extractcol(j));
   return lutb;
   }

/*!
   \brief Object initialization
   Determines code parameters from LUT and sets up object for use.
   This includes validating the LUT, setting up the pilot generator,
   and clearing the pilot sequence.
*/

template <class real, bool norm>
void dminner2d<real,norm>::init()
   {
   // Determine code parameters from LUT
   q = lut.size();
   assertalways(q > 0);
   m = lut(0).ysize();
   n = lut(0).xsize();
   // Validate LUT
   validatelut();
   // Seed the pilot generator and clear the sequence
   r.seed(0);
   pilot.init(0,0);
   }

// description output

template <class real, bool norm>
std::string dminner2d<real,norm>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative 2D DM Inner Code (";
   sout << m << "x" << n << "/" << q << ", ";
   sout << lutname << " codebook)";
   return sout.str();
   }

// object serialization - saving

template <class real, bool norm>
std::ostream& dminner2d<real,norm>::serialize(std::ostream& sout) const
   {
   sout << lutname;
   sout << lut;
   return sout;
   }

// object serialization - loading

template <class real, bool norm>
std::istream& dminner2d<real,norm>::serialize(std::istream& sin)
   {
   sin >> lutname;
   sin >> lut;
   init();
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2d<logrealfast,false>;
template <>
const serializer dminner2d<logrealfast,false>::shelper
   = serializer("blockmodem", "dminner2d<logrealfast>", dminner2d<logrealfast,false>::create);

template class dminner2d<double,true>;
template <>
const serializer dminner2d<double,true>::shelper
   = serializer("blockmodem", "dminner2d<double>", dminner2d<double,true>::create);

}; // end namespace
