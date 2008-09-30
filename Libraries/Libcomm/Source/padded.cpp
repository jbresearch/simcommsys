/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "padded.h"
#include <sstream>

namespace libcomm {

using libbase::vector;
using libbase::matrix;

const libbase::serializer padded::shelper("interleaver", "padded", padded::create);

// construction and destruction

padded::padded()
   {
   otp = NULL;
   inter = NULL;
   }

padded::padded(const interleaver& inter, const fsm& encoder, const int tau, const bool terminated, const bool renewable)
   {
   otp = new onetimepad(encoder, tau, terminated, renewable);
   padded::inter = inter.clone();
   }

padded::padded(const padded& x)
   {
   inter = x.inter->clone();
   otp = x.otp->clone();
   }

padded::~padded()
   {
   if(otp != NULL)
      delete otp;
   if(inter != NULL)
      delete inter;
   }

// inter-frame operations

void padded::seedfrom(libbase::random& r)
   {
   otp->seedfrom(r);
   }

void padded::advance()
   {
   otp->advance();
   }

// transform functions

void padded::transform(const vector<int>& in, vector<int>& out) const
   {
   vector<int> temp;
   inter->transform(in, temp);
   otp->transform(temp, out);
   }

void padded::transform(const matrix<double>& in, matrix<double>& out) const
   {
   matrix<double> temp;
   inter->transform(in, temp);
   otp->transform(temp, out);
   }

void padded::inverse(const matrix<double>& in, matrix<double>& out) const
   {
   matrix<double> temp;
   otp->inverse(in, temp);
   inter->inverse(temp, out);
   }

void padded::transform(const matrix<libbase::logrealfast>& in, matrix<libbase::logrealfast>& out) const
   {
   matrix<libbase::logrealfast> temp;
   inter->transform(in, temp);
   otp->transform(temp, out);
   }

void padded::inverse(const matrix<libbase::logrealfast>& in, matrix<libbase::logrealfast>& out) const
   {
   matrix<libbase::logrealfast> temp;
   otp->inverse(in, temp);
   inter->inverse(temp, out);
   }

// description output

std::string padded::description() const
   {
   std::ostringstream sout;
   sout << "Padded Interleaver [" << inter->description() << " + " << otp->description() << "]";
   return sout.str();
   }

// object serialization - saving

std::ostream& padded::serialize(std::ostream& sout) const
   {
   sout << otp;
   sout << inter;
   return sout;
   }

// object serialization - loading

std::istream& padded::serialize(std::istream& sin)
   {
   sin >> otp;
   sin >> inter;
   return sin;
   }

}; // end namespace
