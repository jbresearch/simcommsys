#ifndef __turbo_h
#define __turbo_h

#include "config.h"
#include "vcs.h"
#include "serializer.h"

#include "codec.h"
#include "fsm.h"
#include "interleaver.h"
#include "bcjr.h"

#include <stdlib.h>
#include <math.h>

/*!
\brief   Class implementing the Turbo decoding algorithm.
\author  Johann Briffa

  All internal metrics are held as type 'real', which is user-defined. This allows internal working
  at any required level of accuracy. This is required because the internal matrics have a very wide
  dynamic range, which increases exponentially with block size 'tau'. Actually, the required range
  is within [1,0), but very large exponents are required. (For BCJR sub-component)
 
  Version 1.10 (4 Mar 1999)
  updated intialisation of a priori statistics (now they are 1/K instead of 1).
  
  Version 1.11 (5 Mar 1999)
  renamed the mem_order() function to tail_length(), which is more indicative of the true
  use of this function (complies with codec 1.01).
   
  Version 1.20 (5 Mar 1999)
  modified tail_length() to return 0 when the turbo codes is defined with endatzero==false.
  This makes the turbo module handle untailed sequences correctly.
    
  Version 1.30 (8 Mar 1999)
  modified to use the faster BCJR decode routine (does not compute output statistics).
     
  Version 1.40 (20 Apr 1999)
  removed the reinitialisation of data probabilities in the tail section for non-simile
  interleavers. The data is always valid input, whether the tails are the same or not.
      
  Version 1.41 (22 Apr 1999)
  made turbo allocate memory on first call to demodulate/decode
       
  Version 1.50 (7 Jun 1999)
  modified the system to comply with codec 1.10.
        
  Version 1.60 (7 Jun 1999)
  revamped handling of puncturing with the use of an additional class.
         
  Version 1.70 (15 Mar 2000)
  changed passing of interleaver from an array to a vector; also removed the
  redundant passing of (tau) [not yet] and sets, since these can be taken from the
  interleaver.
           
  Version 1.80 (15 Mar 2000)
  modified decoder to allow parallel decoding as well as serial (serial slightly
  changed, since order of de-int/int is done differently, but this should not
  change any results)
             
  Version 1.81 (6 Jun 2000)
  reinstated the a posteriori statistics as part of the class rather than the decode
  function. This simplifies the construction of derived classes that make use of this
  information.
               
  Version 1.82 (21 Oct 2001)
  moved most functions to the cpp file rather than the header, in order to be able to
  compile the header with Microsoft Extensions. Naturally compilation is faster, but this
  also requires realizations of the turbo<> class within the cpp file. This was done for
  mpreal, mpgnu and logreal.

  Version 1.83 (3 Nov 2001)
  modified parallel decoder to solve a bug that was crashing the simulator when working
  with parallel codes using more than 2 parallel decoders (ie more than 1 interleaver).
  For stability, when working the new set of a priori information for the next stage, we
  now divide the sum of extrinsic information from all other decoders by the number of 
  elements making it up (that is, in practice we work the average not the sum). This seems
  to be working fine now from preliminary simulations.

  Version 1.84 (4 Nov 2001)
  added a function which outputs details on the codec (in accordance with codec 1.20)

  Version 1.85 (15 Nov 2001)
  added functions which return the code's block size (to be eventually included in all
  codec types), and also the data rate.

  Version 1.90 (15 Nov 2001)
  modified the way that extrinsic information is worked for interleaved codes, in that
  the order of de-int/int and working extrinsic information is returned to what it was
  before Version 1.80 - this should solve the divergence problem I am having with JPL
  termination interleavers. Note that the way this has been reimplemented favors speed
  over memory use (ie I am keeping all interleaved versions of the intrinsic source
  information). I could keep only that for the non-interleaved source vector and re-
  create the others every time.

  Version 1.91 (23 Feb 2002)
  added flushes to all end-of-line clog outputs, to clean up text user interface.

  Version 1.92 (1 Mar 2002)   
  edited the classes to be compileable with Microsoft extensions enabled - in practice,
  the major change is in for() loops, where MS defines scope differently from ANSI.
  Rather than taking the loop variables into function scope, we chose to wrap around the
  offending for() loops.

  Version 1.93 (6 Mar 2002)
  also changed use of iostream from global to std namespace.

  Version 2.00 (13 Mar 2002)
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

  Version 2.10 (18 Mar 2002)
  modified to conform with codec 1.41.
  In the process, removed the use of puncturing from within this class, and also
  the information functions block_inbits, block_outbits and block_rate, since these
  are defined (with different names, though) in codec 1.41. Since puncturing is no
  longer handled within the codec, for the moment, the modification for stipple 
  puncturing with simile interleavers is not performed.
  Also, removed the clog & cerr information output during initialization.

  Version 2.11 (19 Mar 2002)
  made internal data members (and direct manipulation functions) private not protected.
  Also added a protected const function that gives derived classes read access to the
  a posteriori statistics from the decoder (this was necessary for the diffturbo class.

  Version 2.12 (23 Mar 2002)
  changed the definition of shelper - instead of a template definition, which uses the
  typeid.name() function to determine the name of the arithmetic class, we now manually
  define shelper for each instantiation. This avoids the ambiguity (due to implementation
  dependence) in name().

  Version 2.20 (27 Mar 2002)
  changed descriptive output function to conform with codec 1.50.

  Version 2.21 (11 Jun 2002)
  modified the definition of num_outputs() to convert the result to int before returning.

  Version 2.30 (18 Apr 2005)
  added a second template class 'dbl', which defaults to 'double', to allow other
  numerical representations for inter-iteration statistics. This became necessary
  for the parallel decoding structure, where the range of extrinsic information is much
  larger than for serial decoding; furthermore, this range increases with the number of
  iterations performed. This change also necessitated bcjr v2.50.

  Version 2.40 (11 Jul 2006)
  added support for circular encoding/decoding. For now, the flag to indicate circular
  encoding/decoding has been added after the one indicating parallel decoding, and has
  also been defined as an optional flag. Eventually, the flag setting for this class
  needs to be improved. Also, note that earlier serialized files need to be updated to
  include the "circular" variable. In fact, it would be good if the serialization concept
  was improved to cater for versioning.

  Version 2.41 (17 Jul 2006)
  in translate, made explicit conversion of round's output to int, to conform with the
  changes in itfunc 1.07.

  Version 2.42 (21 Jul 2006)
  attempting to solve a presumed bug relating to circular decoding (since performance of
  these codes is not as good as expected; it is also, strangely, worse for large blocks)
  * documented bcjr_wrap
  * added debugging assertions in encode process to verify that the circulation state
  was set up properly.
  * updated init to pass 'circular' flag to bcjr module
  * updated translate to reset the start- and end-state probability tables for the
  BCJR algorithm

  Version 2.43 (25 Jul 2006)
  fixed a (serious) bug in 'encode', where the tail bits are only fixed for the first
  (non-interleaved) sequence; this was introduced in the last few days when the encoding
  process was streamlined to use the same lines of code for the non-interleaved and
  the interleaved data (introduced with circular coding). This should explain the approx
  10% decrease in error-correcting performance for the SPECturbo code.

  Version 2.44 (27 Jul 2006)
  minor update - the description function now also includes the number of iterations.

  Version 2.45 (28 Jul - 1 Aug 2006)
  simulations show that parallel-decoding works well with the 1/4-rate, 3-code, K=3
  (111/101), N=4096 code from divs95b; however, when simulating larger codes (N=8192)
  the system seems to go unstable after a few iterations. Also significantly, similar
  codes with lower rates (1/6 and 1/8) perform _worse_ as the rate decreases. This
  version attempts to fix the problem by removing the extrinsic information scaling
  method.

  Version 2.46 (2 Aug 2006)
  following the addition of normalization within the BCJR alpha and beta metric
  computation routines, a similar approach is adopted here by normalizing:
  * the channel-derived (intrinsic) probabilities 'r' and 'R' in translate
    [note: in this function the a-priori probabilities are now created normalized]
  * the extrinsic probabilities in decode_serial and decode_parallel
  Also modified the description routine to print the names of all interleavers.

  Version 2.47 (3 Aug 2006)
  * modified internal wrapping functions 'work_extrinsic', 'bcjr_wrap', 'hard_decision'
  to indicate within the prototype which parameters are input (by making them const).
  While this should not change any results, it is a forward step to simplify debugging,

  Version 2.48 (6 Oct 2006)
  modified for compatibility with VS .NET 2005:
  * in num_outputs & translate, modified use of pow to avoid ambiguity

  Version 2.50 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 2.51 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]
*/

namespace libcomm {

template <class real, class dbl=double> class turbo : public codec, private bcjr<real,dbl> {
   static const libbase::vcs version;
   static const libbase::serializer shelper;
   static void* create() { return new turbo<real,dbl>; };
private:
   libbase::vector<interleaver *> inter;
   fsm		  *encoder;
   double     rate;
   int		  tau;
   int        sets;
   bool       simile, endatzero, parallel, circular;
   int        iter;
   int		  M, K, N, P;    // # of states, inputs, outputs, parity symbols (respectively)
   int		  m;             // memory order of encoder
   // A Priori statistics (intrinsic source, intrinsic encoded, extrinsic source)
   libbase::vector< libbase::matrix<dbl> > r, R, ra;
   // A Posteriori statistics
   libbase::matrix<dbl> ri;
   // Temporary statistics (interleaved versions of ra and ri)
   libbase::matrix<dbl> rai, rii;
   // memory allocator (for internal use only)
   bool initialised;             // Initially false, becomes true when memory is initialised
   void allocate();
   // wrapping functions
   void work_extrinsic(const libbase::matrix<dbl>& ra, const libbase::matrix<dbl>& ri, const libbase::matrix<dbl>& r, libbase::matrix<dbl>& re);
   void bcjr_wrap(const int set, const libbase::matrix<dbl>& ra, libbase::matrix<dbl>& ri, libbase::matrix<dbl>& re);
   void hard_decision(const libbase::matrix<dbl>& ri, libbase::vector<int>& decoded);
   void decode_serial(libbase::matrix<dbl>& ri);
   void decode_parallel(libbase::matrix<dbl>& ri);
protected:
   double aposteriori(const int t, const int i) const { return ri(t,i); };
   void init();
   void free();
   turbo();
public:
   turbo(const fsm& encoder, const int tau, const libbase::vector<interleaver *>& inter, \
      const int iter, const bool simile, const bool endatzero, const bool parallel=false, const bool circular=false);
   ~turbo() { free(); };

   turbo *clone() const { return new turbo(*this); };		// cloning operation
   const char* name() const { return shelper.name(); };

   void seed(const int s);
   
   void encode(libbase::vector<int>& source, libbase::vector<int>& encoded);
   void translate(const libbase::matrix<double>& ptable);
   void decode(libbase::vector<int>& decoded);
   
   int block_size() const { return tau; };
   int num_inputs() const { return K; };
   int num_outputs() const { return int(K*pow(double(P),sets)); };
   int tail_length() const { return m; };
   int num_iter() const { return iter; };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
