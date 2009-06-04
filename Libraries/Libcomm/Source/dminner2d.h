#ifndef __dminner2d_h
#define __dminner2d_h

#include "config.h"

#include "dminner2.h"
#include "informed_modulator.h"
//#include "fba2.h"
#include "matrix.h"

namespace libcomm {

/*!
   \brief   Iterative 2D implementation of Davey-MacKay Inner Code.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Implements a 2D version of the Davey-MacKay inner code, using iterative
   row/column decoding, where the sparse symbols are now two-dimensional.
*/

template <class real, bool norm>
class dminner2d :
   public informed_modulator<bool,libbase::matrix> {
   //private fba2<real,bool,norm> {
public:
   /*! \name Type definitions */
   typedef libbase::matrix<bool>       array2b_t;
   typedef libbase::vector<double>     array1d_t;
   // @}
private:
   /*! \name User-defined parameters */
   std::string lutname;       //!< name to describe codebook
   libbase::vector<array2b_t> lut; //!< sparsifier LUT, one 'm'x'n' matrix per symbol
   // @}
   /*! \name Pre-computed parameters */
   int      q;                //!< number of symbols in message (input) alphabet
   int      m;                //!< number of rows in sparse (output) symbol
   int      n;                //!< number of columns in sparse (output) symbol
   // @}
   /*! \name Internally-used objects */
   //bsid2d   mychan;           //!< bound channel object
   mutable libbase::randgen r; //!< pilot sequence generator
   mutable array2b_t pilot;   //!< pilot sequence (same size as frame)
   // @}
private:
   // Implementations of channel-specific metrics for fba
   //real R(const int i, const array1b_t& r);
   // Atomic modem operations (private as these should never be used)
   const bool modulate(const int index) const
      { assert("Function should not be used."); return false; };
   const int demodulate(const bool& signal) const
      { assert("Function should not be used."); return 0; };
   const int demodulate(const bool& signal, const libbase::vector<double>& app) const
      { assert("Function should not be used."); return 0; };
protected:
   // Interface with derived classes
   void advance() const;
   void domodulate(const int q, const libbase::matrix<int>& encoded, libbase::matrix<bool>& tx);
   void dodemodulate(const channel<bool,libbase::matrix>& chan, const libbase::matrix<bool>& rx, libbase::matrix<array1d_t>& ptable) {};
   void dodemodulate(const channel<bool,libbase::matrix>& chan, const libbase::matrix<bool>& rx, const libbase::matrix<array1d_t>& app, libbase::matrix<array1d_t>& ptable);

private:
   /*! \name Internal functions */
   void validatelut() const;
   // @}
protected:
   /*! \name Internal functions */
   void init();
   // @}
public:

   /*! \name Class-specific informative functions */
   int get_symbol_rows() const { return m; };
   int get_symbol_cols() const { return n; };
   array2b_t get_symbol(int i) const { return lut(i); };
   // @}

   // Setup functions
   void seedfrom(libbase::random& r) { this->r.seed(r.ival()); };

   // Informative functions
   int num_symbols() const { return q; };
   libbase::size_type<libbase::matrix> output_block_size() const
      { return libbase::size_type<libbase::matrix>(input_block_size().x*n, input_block_size().y*m); };
   double energy() const { return m*n; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(dminner2d);
};

}; // end namespace

#endif
