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
         pacc.init(pin);
         pacc = 1;
         for(int ii=0; ii<m; ii++)
            {
            rx.extractrow(rxvec,i*M+ii);
            rowdec.demodulate(mychan, rxvec, pin, pout);
            pacc *= pout;
            }
         ptable.insertrow(pacc,i);
         }
      // Decode columns
      mychan.set_blocksize(M);
      for(int j=0; j<N; j++)
         {
         ptable.extractcol(pin,j);
         pacc.init(pin);
         pacc = 1;
         for(int jj=0; jj<n; jj++)
            {
            rx.extractcol(rxvec,j*N+jj);
            coldec.demodulate(mychan, rxvec, pin, pout);
            pacc *= pout;
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
