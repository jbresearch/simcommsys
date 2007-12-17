#ifndef __commsys_h
#define __commsys_h

#include "config.h"
#include "experiment.h"
#include "randgen.h"
#include "codec.h"
#include "modulator.h"
#include "puncture.h"
#include "channel.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.00

  Version 1.10 (7 Jun 1999)
  modified the system to comply with codec 1.10.

  Version 1.20 (30 Jul 1999)
  added option to speed up Turbo decoding (by stopping when an iteration does not
  improve the error rate).

  Version 1.21 (26 Aug 1999)
  modified stopping criterion for samples such that sample granularity is just above 0.5s
  based on a timer rather than on the number of symbols transmitted

  Version 1.30 (2 Sep 1999)
  added a hook for clients to know the number of frames simulated in a particular run.

  Version 1.31 (1 Mar 2002)   
  edited the classes to be compileable with Microsoft extensions enabled - in practice,
  the major change is in for() loops, where MS defines scope differently from ANSI.
  Rather than taking the loop variables into function scope, we chose to avoid having
  more than one loop per function, by defining private helper functions (or doing away 
  with them if there are better ways of doing the same operation).

  Version 1.32 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.40 (18 Mar 2002)
  changed constructor to take also the modem and an optional puncturing system, besides
  the already present random source (for generating the source stream), the channel
  model, and the codec. This change was necessitated by the definition of codec 1.41.
  Also removed the extremely hazardous "fast" option for speed enhancement (see discussion
  in my journal last Fall). Finally, changed the sample loop to bail out after 0.5s
  rather than after at least 1000 modulation symbols have been transmitted (records
  above indicate this should have been done already - why not?).

  Version 1.41 (19 Mar 2002)
  changed the class definition a little to make it more easily used as a base class for
  any type of communication system simulation (essentially after noticing that all
  other comm simulation class had several functions that were essentially identical).
  This involved changing the access to the createsource and cycleonce from private to
  protected, and also adding an extra function transmitandreceive() which does the
  common work from encoding through demodulation. Finally, the cycleonce() function is
  now virtual - this is the function that derived classes need to override in order to
  make full use of the common commsys framework. Also, the base experiment is now a
  public base not a public virtual one. Also made data members protected to allow
  easier derivation. Also fixed a bug (that we were not setting the channel Eb anywhere;
  we are now doing that in the constructor).

  Version 1.42 (24 Mar 2002)
  fixed a bug in commsys constructor - was calling punc->rate() even when punc was NULL;
  now we just use the codec rate when punc is undefined.

  Version 1.43 (17 Jul 2006)
  in constructor, made explicit conversion of round's output to int.

  Version 1.44 (25 Jul 2006)
  in transmitandreceive(), moved the modulation line outside the punc decision. Should
  not have any effect on results or speed.

  Version 1.50 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.60 (24 Apr 2007)
  * added serialization facility requirement.
  * added copy constructor

  Version 1.61 (29 Oct 2007)
  * updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]

  Version 1.62 (29 Nov 2007)
  * added getters for internal objects
*/

class commsys : public experiment {
   static const libbase::serializer shelper;
   static void* create() { return new commsys; };
protected:
   // bound objects:
   bool        internallyallocated;
   libbase::randgen     *src;
   codec       *cdc;
   modulator   *modem;
   puncture    *punc;
   channel     *chan;
   // working variables (data heap)
   int  tau, m, N, K, k, iter;
   libbase::vector<int> source, encoded, decoded;
   libbase::vector<sigspace>  signal1, signal2;
   libbase::matrix<double> ptable1, ptable2;
protected:
   void createsource();
   void transmitandreceive();
   virtual void cycleonce(libbase::vector<double>& result);
   void init();
   void clear();
   void free();
   commsys();
public:
   commsys(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel *chan);
   commsys(const commsys& c);
   ~commsys() {};
   
   commsys *clone() const { return new commsys(*this); };      // cloning operation
   const char* name() const { return shelper.name(); };

   int count() const { return 2*iter; };
   void seed(int s);
   void set(double x) { chan->set_snr(x); };
   double get() { return chan->get_snr(); };
   void sample(libbase::vector<double>& result, int& samplecount);

   // component object getters
   const codec     *getcodec() const { return cdc; };
   const modulator *getmodem() const { return modem; };
   const puncture  *getpunc() const { return punc; };
   const channel   *getchan() const { return chan; };

   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
