/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "pacifier.h"
#include "cputimer.h"
#include "randgen.h"
#include "truerand.h"

#include <boost/program_options.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

namespace makesrandom {

/*!
 * \brief Vector-type container for holding LUT
 * This vector container has the following additional abilities:
 * - operator() works like operator[]
 * - serialize() output to any stream
 */
template <class T>
class myvector : public std::vector<T> {
private:
   typedef std::vector<T> Base;
public:
   myvector(const int x = 0) :
      Base(x)
      {
      }
   T& operator()(int x)
      {
      return this->operator[](x);
      }
   const T& operator()(int x) const
      {
      return this->operator[](x);
      }
   std::ostream& serialize(std::ostream& sout)
      {
      typename Base::iterator it;
      for (it = Base::begin(); it != Base::end(); it++)
         sout << *it << std::endl;
      return sout;
      }
};

template <class T>
class mylist;

template <class T>
class pair {
public:
   T value;
   typename mylist<T>::iterator index;
};

/*!
 * \brief Container for S-Random interleaver creation
 * The base container must have the following abilities:
 * - removing an element from the middle with erase()
 * - return number of elements with size()
 * - return true when empty with empty()
 * This container has the following additional abilities:
 * - initializing with numerical sequence
 * - random-access iterator return
 */
template <class T>
class mylist : public std::vector<T> {
private:
   typedef std::vector<T> Base;
public:
   mylist(const int x = 0) :
      Base(x)
      {
      }
   typename Base::iterator getiteratorat(const int x)
      {
      typename Base::iterator it = Base::begin();
      for (int i = 0; i < x; i++)
         it++;
      return it;
      }
   void sequence()
      {
      typename Base::iterator it;
      int i = 0;
      for (it = Base::begin(); it != Base::end(); it++)
         *it = i++;
      }
   mylist<pair<T> > pairup()
      {
      mylist<pair<T> > result(Base::size());
      typename Base::iterator thisit;
      typename mylist<pair<T> >::iterator pairit = result.begin();
      for (thisit = Base::begin(); thisit != Base::end(); thisit++)
         {
         pairit->value = *thisit;
         pairit->index = thisit;
         pairit++;
         }
      return result;
      }
};

//! Check if the last element inserted satisfies the spread criterion

bool satisfiesspread(const myvector<int>& lut, const int n, const int i,
      const int spread)
   {
   for (int j = std::max(0, i - spread); j < i; j++)
      if (abs(lut(j) - n) < spread)
         return false;
   return true;
   }

//! S-Random creation process

myvector<int> create_srandom(const int tau, int& spread, libbase::int32u& seed,
      const int max_attempts)
   {
   // set up time-keepers
   libbase::pacifier p;
   // set up random-number generation
   libbase::truerand trng;
   libbase::randgen seeder;
   seeder.seed(trng.ival());
   libbase::randgen prng;
   // initialize space for results
   myvector<int> lut(tau);

   bool failed = true;
   while (failed)
      {
      std::cerr << "Searching for solution at spread " << spread << std::endl;
      // loop for a number of attempts at the given Spread, then
      // reduce and continue as necessary
      libbase::cputimer tmain("Attempt timer");
      int attempt;
      for (attempt = 0; attempt < max_attempts && failed; attempt++)
         {
         std::cerr << p.update(attempt, max_attempts);
         // re-seed random generator
         seed = seeder.ival();
         prng.seed(seed);
         // set up working variables
         mylist<int> unused(tau);
         unused.sequence();
         // loop to fill all entries in the interleaver - or until we fail
         failed = false;
         for (int i = 0; i < tau && !failed; i++)
            {
            // set up for the current entry
            mylist<pair<int> > untried = unused.pairup();
            // loop for the current entry
            // until we find a suitable value or totally fail trying
            while (!failed)
               {
               // choose a random number from what's left to try
               mylist<pair<int> >::iterator ndx = untried.getiteratorat(
                     prng.ival(untried.size()));
               const int n = ndx->value;
               // if it's no good remove it from the list of options
               if (!satisfiesspread(lut, n, i, spread))
                  {
                  untried.erase(ndx);
                  failed = untried.empty();
                  continue;
                  }
               // if it's a suitable value, insert & mark as used
               else
                  {
                  unused.erase(ndx->index);
                  lut(i) = n;
                  break;
                  }
               }
            }
         }
      // show user how fast we're working
      tmain.stop();
      std::cerr << "Attempts: " << attempt << " in " << tmain << std::endl;
      std::cerr << "Speed: " << double(attempt) / tmain.elapsed()
            << " attempts/sec" << std::endl;
      // if this failed, prepare for the next attempt
      if (failed)
         spread--;
      }

   // stop timers
   std::cerr << p.update(max_attempts, max_attempts);

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

void serialize_interleaver(std::ostream& sout, myvector<int> lut, int tau,
      int spread, libbase::int32u seed, double elapsed)
   {
   sout << "#% Size: " << tau << std::endl;
   sout << "#% Spread: " << spread << std::endl;
   sout << "#% Seed: " << seed << std::endl;
   sout << "# Date: " << libbase::timer::date() << std::endl;
   sout << "# Time taken: " << libbase::timer::format(elapsed) << std::endl;
   lut.serialize(sout);
   }

/*!
 * \brief   S-Random Interleaver Creator
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("tau,t", po::value<int>(), "interleaver length");
   desc.add_options()("spread,s", po::value<int>(),
         "interleaver spread to start with");
   desc.add_options()("attempts,n", po::value<int>()->default_value(1000),
         "number of attempts before reducing spread");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      std::cerr << desc << std::endl;
      return 1;
      }

   // Interpret arguments
   const int tau = vm["tau"].as<int> ();
   int spread = vm["spread"].as<int> ();
   const int max_attempts = vm["attempts"].as<int> ();
   // Main process
   libbase::int32u seed = 0;
   myvector<int> lut = create_srandom(tau, spread, seed, max_attempts);
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
