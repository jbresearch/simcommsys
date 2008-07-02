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

   All internal metrics are held as type 'real', which is user-defined. This allows internal working
   at any required level of accuracy. This is required because the internal matrics have a very wide
   dynamic range, which increases exponentially with block size 'tau'. Actually, the required range
   is within [1,0), but very large exponents are required. (For BCJR sub-component)

   \version 1.10 (4 Mar 1999)
   updated intialisation of a priori statistics (now they are 1/K instead of 1).

   \version 1.11 (5 Mar 1999)
   renamed the mem_order() function to tail_length(), which is more indicative of the true
   use of this function (complies with codec 1.01).

   \version 1.20 (5 Mar 1999)
   modified tail_length() to return 0 when the turbo codes is defined with endatzero==false.
   This makes the turbo module handle untailed sequences correctly.

   \version 1.30 (8 Mar 1999)
   modified to use the faster BCJR decode routine (does not compute output statistics).

   \version 1.40 (20 Apr 1999)
   removed the reinitialisation of data probabilities in the tail section for non-simile
   interleavers. The data is always valid input, whether the tails are the same or not.

   \version 1.41 (22 Apr 1999)
   made turbo allocate memory on first call to demodulate/decode

   \version 1.50 (7 Jun 1999)
   modified the system to comply with codec 1.10.

   \version 1.60 (7 Jun 1999)
   revamped handling of puncturing with the use of an additional class.

   \version 1.70 (15 Mar 2000)
   changed passing of interleaver from an array to a vector; also removed the
   redundant passing of (tau) [not yet] and sets, since these can be taken from the
   interleaver.

   \version 1.80 (15 Mar 2000)
   modified decoder to allow parallel decoding as well as serial (serial slightly
   changed, since order of de-int/int is done differently, but this should not
   change any results)

   \version 1.81 (6 Jun 2000)
   reinstated the a posteriori statistics as part of the class rather than the decode
   function. This simplifies the construction of derived classes that make use of this
   information.

   \version 1.82 (21 Oct 2001)
   moved most functions to the cpp file rather than the header, in order to be able to
   compile the header with Microsoft Extensions. Naturally compilation is faster, but this
   also requires realizations of the turbo<> class within the cpp file. This was done for
   mpreal, mpgnu and logreal.

   \version 1.83 (3 Nov 2001)
   modified parallel decoder to solve a bug that was crashing the simulator when working
   with parallel codes using more than 2 parallel decoders (ie more than 1 interleaver).
   For stability, when working the new set of a priori information for the next stage, we
   now divide the sum of extrinsic information from all other decoders by the number of
   elements making it up (that is, in practice we work the average not the sum). This seems
   to be working fine now from preliminary simulations.

   \version 1.84 (4 Nov 2001)
   added a function which outputs details on the codec (in accordance with codec 1.20)

   \version 1.85 (15 Nov 2001)
   added functions which return the code's block size (to be eventually included in all
   codec types), and also the data rate.

   \version 1.90 (15 Nov 2001)
   modified the way that extrinsic information is worked for interleaved codes, in that
   the order of de-int/int and working extrinsic information is returned to what it was
   before Version 1.80 - this should solve the divergence problem I am having with JPL
   termination interleavers. Note that the way this has been reimplemented favors speed
   over memory use (ie I am keeping all interleaved versions of the intrinsic source
   information). I could keep only that for the non-interleaved source vector and re-
   create the others every time.

   \version 1.91 (23 Feb 2002)
   added flushes to all end-of-line clog outputs, to clean up text user interface.

   \version 1.92 (1 Mar 2002)
   edited the classes to be compileable with Microsoft extensions enabled - in practice,
   the major change is in for() loops, where MS defines scope differently from ANSI.
   Rather than taking the loop variables into function scope, we chose to wrap around the
   offending for() loops.

   \version 1.93 (6 Mar 2002)
   also changed use of iostream from global to std namespace.

   \version 2.00 (13 Mar 2002)
   updated the system to conform with the completed serialization protocol (in conformance
   with codec 1.30), by adding the necessary name() function, and also by adding a static
   serializer member and initialize it with this class's name and the static constructor
   (adding that too, together with the necessary protected default constructor). Also made
   the codec object a public base class, rather than a virtual public one, since this
   was affecting the transfer of virtual functions within the class (causing access
   violations). Also made the version info a const member variable, and made all parameters
   of the constructor const; the constructor is then expected to clone all objects and
   the destructor is expected to delete them. The only exception here is that the set
   of interleavers is passed as a vector of pointers. These must be allocated on the heap
   by the user, and will then be deleted by this class. Also added a function which
   deallocates all heap variables.

   \version 2.10 (18 Mar 2002)
   modified to conform with codec 1.41.
   In the process, removed the use of puncturing from within this class, and also
   the information functions block_inbits, block_outbits and block_rate, since these
   are defined (with different names, though) in codec 1.41. Since puncturing is no
   longer handled within the codec, for the moment, the modification for stipple
   puncturing with simile interleavers is not performed.
   Also, removed the clog & cerr information output during initialization.

   \version 2.11 (19 Mar 2002)
   made internal data members (and direct manipulation functions) private not protected.
   Also added a protected const function that gives derived classes read access to the
   a posteriori statistics from the decoder (this was necessary for the diffturbo class.

   \version 2.12 (23 Mar 2002)
   changed the definition of shelper - instead of a template definition, which uses the
   typeid.name() function to determine the name of the arithmetic class, we now manually
   define shelper for each instantiation. This avoids the ambiguity (due to implementation
   dependence) in name().

   \version 2.20 (27 Mar 2002)
   changed descriptive output function to conform with codec 1.50.

   \version 2.21 (11 Jun 2002)
   modified the definition of num_outputs() to convert the result to int before returning.

   \version 2.30 (18 Apr 2005)
   added a second template class 'dbl', which defaults to 'double', to allow other
   numerical representations for inter-iteration statistics. This became necessary
   for the parallel decoding structure, where the range of extrinsic information is much
   larger than for serial decoding; furthermore, this range increases with the number of
   iterations performed. This change also necessitated bcjr v2.50.

   \version 2.40 (11 Jul 2006)
   added support for circular encoding/decoding. For now, the flag to indicate circular
   encoding/decoding has been added after the one indicating parallel decoding, and has
   also been defined as an optional flag. Eventually, the flag setting for this class
   needs to be improved. Also, note that earlier serialized files need to be updated to
   include the "circular" variable. In fact, it would be good if the serialization concept
   was improved to cater for versioning.

   \version 2.41 (17 Jul 2006)
   in translate, made explicit conversion of round's output to int, to conform with the
   changes in itfunc 1.07.

   \version 2.42 (21 Jul 2006)
   attempting to solve a presumed bug relating to circular decoding (since performance of
   these codes is not as good as expected; it is also, strangely, worse for large blocks)
   - documented bcjr_wrap
   - added debugging assertions in encode process to verify that the circulation state
   was set up properly.
   - updated init to pass 'circular' flag to bcjr module
   - updated translate to reset the start- and end-state probability tables for the
   BCJR algorithm

   \version 2.43 (25 Jul 2006)
   fixed a (serious) bug in 'encode', where the tail bits are only fixed for the first
   (non-interleaved) sequence; this was introduced in the last few days when the encoding
   process was streamlined to use the same lines of code for the non-interleaved and
   the interleaved data (introduced with circular coding). This should explain the approx
   10% decrease in error-correcting performance for the SPECturbo code.

   \version 2.44 (27 Jul 2006)
   minor update - the description function now also includes the number of iterations.

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
   Also modified the description routine to print the names of all interleavers.

   \version 2.47 (3 Aug 2006)
   - modified internal wrapping functions 'work_extrinsic', 'bcjr_wrap', 'hard_decision'
   to indicate within the prototype which parameters are input (by making them const).
   While this should not change any results, it is a forward step to simplify debugging,

   \version 2.48 (6 Oct 2006)
   modified for compatibility with VS .NET 2005:
   - in num_outputs & translate, modified use of pow to avoid ambiguity

   \version 2.50 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 2.51 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type.
     [cf. Stroustrup 15.6.2]

   \version 2.52 (2 Jan 2008)
   - made check for correct termination fatal
   - made check for circulation state occur in all builds (was debug only)
   - removed support for simile interleavers

   \version 2.60 (6 Jan 2008)
   - removed various redundant blocks, a remnant from old VS
   - updated to cater for changes in bcjr 2.60; this required the addition of new
     arrays to hold the intermediate values of start- and end-state probabilities
     for circular trellises

   \version 2.70 (16 Apr 2008)
   - added interleaver before the first encoder
   - added version-control for serialization; for compatibility, earlier versions
     are interpreted as v.0; a flat interleaver is automatically used for the
     first encoder in these cases.
   
   \version 2.71 (17 Apr 2008)
   - Removed 'rate', as this was never used
   - Replaced stream-input of bools with direct form
   - Replaced 'm' with tail_length() and moved computation there
   - Replaced 'M' with enc_states() and moved computation there
   - Replaced 'K' with num_inputs() and moved computation there
   - Replaced 'N' with enc_outputs() and moved computation there
   - Replaced 'P' with enc_parity() and moved computation there
   - Replaced 'sets' with num_sets() and moved computation there
   
   \version 2.72 (18 Apr 2008)
   - Replaced loop data-setting and computation with vector/matrix form where
     possible (mostly decode_parallel).
   - Hid the few trace printouts so they only get compiled-in on debug runs
     (avoids evaluating what would have been printed, in some cases this would
     have involved various array/matrix computations)
   - Removed unnecessary conditional in work_extrinsic
   - Replaced error reporting in translate() with assertions
   - Updated bcjr_wrap to perform extrinsic computation after de-interleaving;
     this removes the need for pre-interleaved r() sets in this function
   - Removed pre-interleaved r() set

   \version 2.73 (24 Apr 2008)
   - replaced serialization support with macros

   \version 2.74 (25 Apr 2008)
   - implemented output alphabet size getter

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
