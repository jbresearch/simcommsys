/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "channel.h"

namespace libcomm {

// constructors / destructors

channel<sigspace>::channel()
   {
   Eb = 1;
   set_parameter(0);
   }

// setting and getting overall channel SNR

void channel<sigspace>::compute_noise()
   {
   No = 0.5 * pow(10.0, -snr_db / 10.0);
   // call derived class handle
   compute_parameters(Eb, No);
   }

void channel<sigspace>::set_eb(const double Eb)
   {
   channel::Eb = Eb;
   compute_noise();
   }

void channel<sigspace>::set_no(const double No)
   {
   snr_db = -10.0 * log10(2 * No);
   compute_noise();
   }

void channel<sigspace>::set_parameter(const double snr_db)
   {
   channel::snr_db = snr_db;
   compute_noise();
   }

} // end namespace
