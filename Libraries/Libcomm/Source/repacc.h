#ifndef __repacc_h
#define __repacc_h

#include "config.h"
#include "codec_softout.h"
#include "fsm.h"
#include "interleaver.h"
#include "bcjr.h"

namespace libcomm {

/*!
   \brief   Repeat-Accumulate (RA) codes.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   These codes are decoded using the MAP decoder, rather than the
   sum-product algorithm.

   \todo Implement repeater as fsm

   \todo Avoid divisions when computing extrinsic information
*/

template <class real, class dbl=double>
class repacc : public codec_softout<dbl>, protected bcjr<real,dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int>        array1i_t;
   typedef libbase::vector<dbl>        array1d_t;
   typedef libbase::matrix<dbl>        array2d_t;
   typedef libbase::vector<array1d_t>  array1vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef repacc<real,dbl> This;
   typedef bcjr<real,dbl> BCJR;
private:
   /*! \name User-defined parameters */
   //! Interleaver between repeater and accumulator
   interleaver<dbl> *inter;
   fsm      *encoder;      //!< Encoder representation of accumulator
   int      N;             //!< Block size in input symbols
   int      r;             //!< Repetition factor
   int      iter;          //!< Number of iterations to perform
   bool     endatzero;     //!< Flag to indicate that trellises are terminated
   // @}
protected:
   /*! \name Internal object representation */
   bool     initialised;   //!< Flag to indicate when memory is initialised
   array2d_t rp;           //!< Intrinsic source statistics (natural)
   array2d_t ra;           //!< Extrinsic accumulator-input statistics (natural)
   array2d_t R;            //!< Intrinsic accumulator-output statistics (interleaved)
   // @}
protected:
   /*! \name Internal functions */
   void init();
   void free();
   void reset();
   //! Memory allocator (for internal use only)
   void allocate();
   // @}
public:
   /*! \name Constructors / Destructors */
   repacc();
   ~repacc() { free(); };
   // @}

   // Codec operations
   void seedfrom(libbase::random& r);
   void encode(const array1i_t& source, array1i_t& encoded);
   void translate(const libbase::vector< libbase::vector<double> >& ptable);
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size<libbase::vector> input_block_size() const
      { return libbase::size<libbase::vector>(N); };
   libbase::size<libbase::vector> output_block_size() const
      { return libbase::size<libbase::vector>(N*r + tail_length()); };
   int num_inputs() const { return encoder->num_inputs(); };
   int num_outputs() const { return encoder->num_outputs()/encoder->num_inputs(); };
   int tail_length() const { return endatzero ? encoder->mem_order() : 0; };
   int num_iter() const { return iter; };

   /*! \name Codec information functions - internal */
   int num_repeats() const { return r; };
   const interleaver<dbl> *get_inter() const { return inter; };
   // @}

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(repacc);
};

}; // end namespace

#endif
