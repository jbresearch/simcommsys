#include "serializer_libcomm.h"
#include "commsys.h"
#include "truerand.h"
#include "pacifier.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

namespace makesrandom {

/*!
 \brief Vector-type container for S-Random interleaver creation
 This vector container has the following addional abilities:
 * removing an element from the middle
 * initializing with numerical sequence
 */
template <class T>
class DerivedVector : public libbase::vector<T> {
private:
   typedef libbase::vector<T> Base;
   using Base::m_size;
   using Base::m_data;
public:
   DerivedVector(const int x = 0) :
      Base(x)
      {
      }
   void remove(const int x);
   void sequence();
};

template <class T>
void DerivedVector<T>::remove(const int x)
   {
   assert(x < m_size.length());
   for (int i = x; i < m_size.length() - 1; i++)
      m_data[i] = m_data[i + 1];
   m_size = libbase::size_type<libbase::vector>(m_size.length() - 1);
   }

template <class T>
void DerivedVector<T>::sequence()
   {
   for (int i = 0; i < m_size.length(); i++)
      m_data[i] = i;
   }

//! S-Random creation process

libbase::vector<int> create_srandom(const int tau, int& spread,
      libbase::int32u& seed, const int max_attempts)
   {
   // set up common elements
   libbase::truerand trng;
   libbase::randgen seeder;
   seeder.seed(trng.ival());
   libbase::randgen prng;
   libbase::pacifier p;
   libbase::vector<int> lut(tau);

   bool failed;
   int attempt = 0;
   // loop for a number of attempts at the given Spread, then
   // reduce and continue as necessary
   do
      {
      std::cerr << p.update(attempt, max_attempts);
      // re-seed random generator
      seed = seeder.ival();
      prng.seed(seed);
      // set up working variables
      DerivedVector<int> unused(tau);
      unused.sequence();
      // loop to fill all entries in the interleaver - or until we fail
      failed = false;
      for (int i = 0; i < tau && !failed; i++)
         {
         // set up for the current entry
         DerivedVector<int> untried = unused;
         DerivedVector<int> index(unused.size());
         index.sequence();
         int n, ndx;
         bool good;
         // loop for the current entry - until we manage to find a suitable value
         // or totally fail in trying
         do
            {
            // choose a random number from what's left to try
            ndx = prng.ival(untried.size());
            n = untried(ndx);
            // see if it's a suitable value (ie satisfies spread constraint)
            good = true;
            for (int j = std::max(0, i - spread); j < i; j++)
               if (abs(lut(j) - n) < spread)
                  {
                  good = false;
                  break;
                  }
            // if it's no good remove it from the list of options,
            // if it's good then insert it into the interleaver & mark that number as used
            if (!good)
               {
               untried.remove(ndx);
               index.remove(ndx);
               failed = (untried.size() == 0);
               }
            else
               {
               unused.remove(index(ndx));
               lut(i) = n;
               }
            } while (!good && !failed);
         }
      // if this failed, prepare for the next attempt
      if (failed)
         {
         attempt++;
         if (attempt >= max_attempts)
            {
            attempt = 0;
            spread--;
            std::cerr << p.update(attempt, max_attempts);
            std::cerr << "Searching for solution at spread " << spread << "\n";
            }
         }
      } while (failed);

   return lut;
   }

//! Returns filename according to usual convention

std::string compose_filename(int tau, int spread, libbase::int32u seed)
   {
   std::ostringstream sout;
   sout << "sri-" << tau << "-spread" << spread << "-seed" << seed << ".txt";
   return sout.str();
   }

//! Saves the interleaver to the given stream

void serialize_interleaver(std::ostream& sout, libbase::vector<int> lut,
      int tau, int spread, libbase::int32u seed, double elapsed)
   {
   sout << "#% Size: " << tau << "\n";
   sout << "#% Spread: " << spread << "\n";
   sout << "#% Seed: " << seed << "\n";
   sout << "# Date: " << libbase::timer::date() << "\n";
   sout << "# Time taken: " << libbase::timer::format(elapsed) << "\n";
   lut.serialize(sout, '\n');
   }

/*!
 \brief   S-Random Interleaver Creator
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

int main(int argc, char *argv[])
   {
   libbase::timer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message")("tau,t",
         po::value<int>(), "interleaver length")("spread,s", po::value<int>(),
         "interleaver spread to start with")("attempts,n",
         po::value<int>()->default_value(1000),
         "number of attempts before reducing spread");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      std::cerr << desc << "\n";
      return 1;
      }

   // Interpret arguments
   const int tau = vm["tau"].as<int> ();
   int spread = vm["spread"].as<int> ();
   const int max_attempts = vm["attempts"].as<int> ();
   // Main process
   libbase::int32u seed = 0;
   libbase::vector<int> lut = create_srandom(tau, spread, seed, max_attempts);
   // Output
   const std::string fname = compose_filename(tau, spread, seed);
   std::ofstream file(fname.c_str());
   serialize_interleaver(file, lut, tau, spread, seed, tmain.elapsed());

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return makesrandom::main(argc, argv);
   }
