#ifndef __turbo_h
#define __turbo_h

#include "config.h"
#include "serializer.h"

#include "codec.h"
#include "fsm.h"
#include "interleaver.h"
#include "bcjr.h"
#include "itfunc.h"

#include <stdlib.h>
#include <math.h>

namespace libcomm {

/*!
   \brief   Turbo decoding algorithm.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   All internal metrics are held as type 'real', which is user-defined. This
   allows internal working at any required level of accuracy. This is necessary
   because the internal matrics have a very wide dynamic range, which increases
   exponentially with block size 'tau'. Actually, the required range is within
   [1,0), but very large exponents are required. (For BCJR sub-component)

   \note Memory is allocaed only on first call to demodulate/decode. This
         reduces memory requirements in cases where classes are instantiated
         but not actually used for decoding (e.g. in master node on a
         distributed Monte Carlo simulation)

   \note Since puncturing is not handled within the codec, for the moment, the
         modification for stipple puncturing with simile interleavers is not
         performed.

   \note The template class 'dbl', which defaults to 'double', defines the
         numerical representation for inter-iteration statistics. This became
         necessary for the parallel decoding structure, where the range of
         extrinsic information is much larger than for serial decoding;
         furthermore, this range increases with the number of iterations
         performed.

   \version 1.10 (4 Mar 1999)
   updated intialisation of a priori statistics (now they are 1/K instead of 1).

   \version 1.40 (20 Apr 1999)
   removed the reinitialisation of data probabilities in the tail section for non-simile
   interleavers. The data is always valid input, whether the tails are the same or not.

   \version 1.81 (6 Jun 2000)
   reinstated the a posteriori statistics as part of the class rather than the decode
   function. This simplifies the construction of derived classes that make use of this
   information.

   \version 1.83 (3 Nov 2001)
   modified parallel decoder to solve a bug that was crashing the simulator when working
   with parallel codes using more than 2 parallel decoders (ie more than 1 interleaver).
   For stability, when working the new set of a priori information for the next stage, we
   now divide the sum of extrinsic information from all other decoders by the number of
   elements making it up (that is, in practice we work the average not the sum). This seems
   to be working fine now from preliminary simulations.

   \version 1.90 (15 Nov 2001)
   modified the way that extrinsic information is worked for interleaved codes, in that
   the order of de-int/int and working extrinsic information is returned to what it was
   before Version 1.80 - this should solve the divergence problem I am having with JPL
   termination interleavers. Note that the way this has been reimplemented favors speed
   over memory use (ie I am keeping all interleaved versions of the intrinsic source
   information). I could keep only that for the non-interleaved source vector and re-
   create the others every time.

   \version 2.40 (11 Jul 2006)
   added support for circular encoding/decoding. For now, the flag to indicate circular
   encoding/decoding has been added after the one indicating parallel decoding, and has
   also been defined as an optional flag. Eventually, the flag setting for this class
   needs to be improved. Also, note that earlier serialized files need to be updated to
   include the "circular" variable. In fact, it would be good if the serialization concept
   was improved to cater for versioning.

   \version 2.43 (25 Jul 2006)
   fixed a (serious) bug in 'encode', where the tail bits are only fixed for the first
   (non-interleaved) sequence; this was introduced in the last few days when the encoding
   process was streamlined to use the same lines of code for the non-interleaved and
   the interleaved data (introduced with circular coding). This should explain the approx
   10% decrease in error-correcting performance for the SPECturbo code.

   \version 2.45 (28 Jul - 1 Aug 2006)
   simulations show that parallel-decoding works well with the 1/4-rate, 3-code, K=3
   (111/101), N=4096 code from divs95b; however, when simulating larger codes (N=8192)
   the system seems to go unstable after a few iterations. Also significantly, similar
   codes with lower rates (1/6 and 1/8) perform _worse_ as the rate decreases. This
   version attempts to fix the problem by removing the extrinsic information scaling
   method.

   \version 2.46 (2 Aug 2006)
   following the addition of normalization within the BCJR alpha and beta metric
   computation routines, a similar approach is adopted here by normalizing:
   - the channel-derived (intrinsic) probabilities 'r' and 'R' in translate
    [note: in this function the a-priori probabilities are now created normalized]
   - the extrinsic probabilities in decode_serial and decode_parallel

   \version 2.70 (16 Apr 2008)
   - added interleaver before the first encoder
   - added version-control for serialization; for compatibility, earlier versions
     are interpreted as v.0; a flat interleaver is automatically used for the
     first encoder in these cases.
   
   \version 2.72 (18 Apr 2008)
   - Hid the few trace printouts so they only get compiled-in on debug runs
     (avoids evaluating what would have been printed, in some cases this would
     have involved various array/matrix computations)

   \todo
   - Remove tau from user parameters, as this can be derived from interleavers
     (requires a change to interleaver interface)
   - Fix terminated sequence encoding (implicitly assume a flat first interleaver)
   - Move temporary matrix in translate() to a class member (consider if this
     will actually constitute a speedup)
   - Standardize encoding/decoding of multiple symbols within a larger symbol
     space; this parallels what was done in ccfsm.
*/

template <class real, class dbl=double>
class turbo : public codec, private bcjr<real,dbl> {
private:
   /*! \name User-defined parameters */
   libbase::vector<interleaver *> inter;     //!< Set of interleavers, one per parity sequence
   fsm      *encoder;      //!< Encoder object (same for all parity sequences)
   int      tau;           //!< Length of interleavers (information sequence + tail)
   int      iter;          //!< Number of iterations to perform
   bool     endatzero;     //!< Flag to indicate that trellises are terminated
   bool     parallel;      //!< Flag to enable parallel decoding algorithm (rather than serial)
   bool     circular;      //!< Flag to indicate trellis tailbiting
   // @}
   /*! \name Internal object representation */
   bool initialised;       //!< Initially false, becomes true when memory is initialised
   libbase::matrix<dbl> rp;   //!< A priori intrinsic source statistics (natural)
   libbase::matrix<dbl> ri;   //!< A posteriori source statistics (natural)
   libbase::vector< libbase::matrix<dbl> > R;   //!< A priori intrinsic encoder-output statistics (interleaved)
   libbase::vector< libbase::matrix<dbl> > ra;  //!< A priori extrinsic source statistics
   libbase::vector< libbase::vector<dbl> > ss;  //!< Holder for start-state probabilities (used with circular trellises)
   libbase::vector< libbase::vector<dbl> > se;  //!< Holder for end-state probabilities (used with circular trellises)
   // @}
   /*! \name Temporary variables */
   libbase::matrix<dbl> rai;  //!< Temporary statistics (interleaved version of ra)
   libbase::matrix<dbl> rii;  //!< Temporary statistics (interleaved version of ri)
   // @}
   /*! \name Internal functions */
   //! Memory allocator (for internal use only)
   void allocate();
   // wrapping functions
   void work_extrinsic(const libbase::matrix<dbl>& ra, const libbase::matrix<dbl>& ri, const libbase::matrix<dbl>& r, libbase::matrix<dbl>& re);
   void bcjr_wrap(const int set, const libbase::matrix<dbl>& ra, libbase::matrix<dbl>& ri, libbase::matrix<dbl>& re);
   void hard_decision(const libbase::matrix<dbl>& ri, libbase::vector<int>& decoded);
   void decode_serial(libbase::matrix<dbl>& ri);
   void decode_parallel(libbase::matrix<dbl>& ri);
   // @}
protected:
   /*! \name Internal functions */
   /*! \brief Read access for a-posteriori statistics
      This access is available only for derived classes, and was initially
      added for use by the diffturbo class.
   */
   double aposteriori(const int t, const int i) const { return ri(t,i); };
   void init();
   void free();
   void reset();
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   turbo();
   // @}
public:
   /*! \name Constructors / Destructors */
   turbo(const fsm& encoder, const int tau, const libbase::vector<interleaver *>& inter, \
      const int iter, const bool endatzero, const bool parallel=false, const bool circular=false);
   ~turbo() { free(); };
   // @}

   // Codec operations
   void seedfrom(libbase::random& r);
   void encode(libbase::vector<int>& source, libbase::vector<int>& encoded);
   void translate(const libbase::matrix<double>& ptable);
   void decode(libbase::vector<int>& decoded);

   // Codec information functions - fundamental
   int block_size() const { return tau; };
   int num_inputs() const { return encoder->num_inputs(); };
   int num_outputs() const { return int(num_inputs()*pow(enc_parity(),num_sets())); };
   int output_alphabet() const { return libbase::gcd(num_inputs(),enc_parity()); };
   int tail_length() const { return endatzero ? encoder->mem_order() : 0; };
   int num_iter() const { return iter; };

   /*! \name Codec information functions - internal */
   int num_sets() const { return inter.size(); };
   int enc_states() const { return encoder->num_states(); };
   int enc_outputs() const { return encoder->num_outputs(); };
   int enc_parity() const { return enc_outputs()/num_inputs(); };
   // @}

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(turbo)
};

}; // end namespace

#endif
