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

#include "montecarlo.h"

#include "version.h"
#include "fsm.h"
#include "itfunc.h"
#include "randgen.h"
#include <sstream>
#include <limits>

namespace libcomm {

using std::cerr;
using libbase::trace;
using libbase::vector;

// worker processes

void montecarlo::slave_getcode(void)
   {
   delete system;
   // Receive system as a string
   std::string systemstring;
   if (!receive(systemstring))
      exit(1);
   // Create system object from serialization
   std::istringstream is(systemstring);
   is >> system;
   // Compute its digest
   is.seekg(0);
   sysdigest.process(is);
   // Tell the user what we've done
   cerr << "Date: " << libbase::timer::date() << std::endl;
   cerr << system->description() << std::endl;
   cerr << "Digest: " << std::string(sysdigest) << std::endl;
   }

void montecarlo::slave_getparameter(void)
   {
   cerr << "Date: " << libbase::timer::date() << std::endl;

   seed_experiment();
   double x;
   if (!receive(x))
      exit(1);
   system->set_parameter(x);

   cerr << "Simulating system at parameter = " << system->get_parameter()
         << std::endl;
   }

void montecarlo::slave_work(void)
   {
   // Initialise running values
   system->reset();

   // Iterate for 500ms, which is a good compromise between efficiency and usability
   libbase::walltimer tslave("montecarlo_slave");
   while (tslave.elapsed() < 0.5)
      sampleandaccumulate();
   tslave.stop(); // to avoid expiry

   // Send system digest and current parameter back to master
   assertalways(send(sysdigest));
   assertalways(send(system->get_parameter()));
   // Send accumulated results back to master
   libbase::vector<double> state;
   system->get_state(state);
   assertalways(send(system->get_samplecount()));
   assertalways(send(state));

   // print something to inform the user of our progress
   vector<double> result, errormargin;
   updateresults(result, errormargin);
   display(result, errormargin);
   }

// helper functions

std::string montecarlo::get_systemstring()
   {
   std::ostringstream os;
   os << system;
   std::string systemstring = os.str();
   return systemstring;
   }

/*! \brief Seed the random generators in the experiment
 *
 * Use the stored seed to initialize a PRNG for seeding the embedded system.
 */
void montecarlo::seed_experiment()
   {
   libbase::randgen prng;
   prng.seed(seed);
   system->seedfrom(prng);
   cerr << "Seed: " << seed << std::endl;
   }

void montecarlo::createfunctors(void)
   {
   fgetcode = new libbase::specificfunctor<montecarlo>(this,
         &libcomm::montecarlo::slave_getcode);
   fgetparameter = new libbase::specificfunctor<montecarlo>(this,
         &libcomm::montecarlo::slave_getparameter);
   fwork = new libbase::specificfunctor<montecarlo>(this,
         &libcomm::montecarlo::slave_work);
   // register functions
   fregister("slave_getcode", fgetcode);
   fregister("slave_getparameter", fgetparameter);
   fregister("slave_work", fwork);
   }

void montecarlo::destroyfunctors(void)
   {
   delete fgetcode;
   delete fgetparameter;
   delete fwork;
   }

// System-specific file-handler functions

void montecarlo::writeheader(std::ostream& sout) const
   {
   assert(sout.good());
   assert(system != NULL);
   trace << "DEBUG (montecarlo): writing results header." << std::endl;
   // Print information on the simulation being performed
   trace << "DEBUG (montecarlo): position before = " << sout.tellp()
         << std::endl;
   sout << "#% " << system->description() << std::endl;
   sout << "#% Confidence Level: " << get_confidence_level() << std::endl;
   sout << "#% Convergence Mode: " << get_convergence_mode() << std::endl;
   sout << "#% Date: " << libbase::timer::date() << std::endl;
   sout << "#% Build: " << SIMCOMMSYS_BUILD << std::endl;
   sout << "#% Version: " << SIMCOMMSYS_VERSION << std::endl;
   sout << "#" << std::endl;
   // Print results header
   sout << "# Par";
   for (int i = 0; i < system->count(); i++)
      sout << "\t" << system->result_description(i) << "\tTol";
   sout << "\tSamples\tCPUtime" << std::endl;
   trace << "DEBUG (montecarlo): position after = " << sout.tellp()
         << std::endl;
   }

void montecarlo::writeresults(std::ostream& sout,
      libbase::vector<double>& result, libbase::vector<double>& errormargin) const
   {
   assert(sout.good());
   if (get_samplecount() == 0)
      return;
   trace << "DEBUG (montecarlo): writing results." << std::endl;
   // Write current estimates to file
   trace << "DEBUG (montecarlo): position before = " << sout.tellp()
         << std::endl;
   sout << system->get_parameter();
   for (int i = 0; i < system->count(); i++)
      sout << '\t' << result(i) << '\t' << errormargin(i);
   sout << '\t' << get_samplecount();
   sout << '\t' << getcputime() << std::endl;
   trace << "DEBUG (montecarlo): position after = " << sout.tellp()
         << std::endl;
   }

void montecarlo::writestate(std::ostream& sout) const
   {
   assert(sout.good());
   if (get_samplecount() == 0)
      return;
   trace << "DEBUG (montecarlo): writing state." << std::endl;
   // Write accumulated values to file
   trace << "DEBUG (montecarlo): position before = " << sout.tellp()
         << std::endl;
   libbase::vector<double> state;
   system->get_state(state);
   sout << "## System: " << sysdigest << std::endl;
   sout << "## Parameter: " << system->get_parameter() << std::endl;
   sout << "## Samples: " << get_samplecount() << std::endl;
   sout << "## State: " << state.size() << '\t';
   state.serialize(sout, '\t');
   sout << std::flush;
   trace << "DEBUG (montecarlo): position after = " << sout.tellp()
         << std::endl;
   }

void montecarlo::lookforstate(std::istream& sin)
   {
   assert(sin.good());
   // state variables to read
   std::string digest;
   double parameter = 0;
   libbase::int64u samplecount = 0;
   vector<double> state;
   // read through entire file
   trace << "DEBUG (montecarlo): looking for state." << std::endl;
   sin.seekg(0);
   while (!sin.eof())
      {
      std::string s;
      getline(sin, s);
      if (s.substr(0, 10) == "## System:")
         digest = s.substr(10);
      else if (s.substr(0, 13) == "## Parameter:")
         std::istringstream(s.substr(13)) >> parameter;
      else if (s.substr(0, 11) == "## Samples:")
         std::istringstream(s.substr(11)) >> samplecount;
      else if (s.substr(0, 9) == "## State:")
         {
         std::istringstream is(s.substr(9));
         is >> state;
         }
      }
   // reset file
   sin.clear();
   // check that results correspond to system under simulation
   if (digest == std::string(sysdigest) && parameter == system->get_parameter())
      {
      cerr << "NOTICE: Reloading state with " << samplecount << " samples."
            << std::endl;
      system->accumulate_state(samplecount, state);
      }
   }

// overrideable user-interface functions

/*!
 * \brief Default progress display routine.
 *
 * \note Display updates are rate-limited
 */
void montecarlo::display(const libbase::vector<double>& result,
      const libbase::vector<double>& errormargin) const
   {
   if (tupdate.elapsed() > 0.5)
      {
      using std::clog;
      const std::streamsize prec = clog.precision(3);
      clog << "Timer: " << t << ", ";
      if (isenabled())
         clog << getnumslaves() << " clients, ";
      else
         clog << "local, ";
      clog << getcputime() / t.elapsed() << "× usage, ";
      clog << "pass " << system->get_samplecount() << "." << std::endl;
      clog << "System parameter: " << system->get_parameter() << std::endl;
      clog << "Results:" << std::endl;
      system->prettyprint_results(clog, result, errormargin);
      clog << "Press 'q' to interrupt." << std::endl;
      clog.precision(prec);
      tupdate.start();
      }
   }

// main process

/*!
 * \brief Determine overall estimate from accumulated results
 * \param[out] result      Vector containing the set of estimates
 * \param[out] errormargin Corresponding margin of error (radius of confidence interval)
 */
void montecarlo::updateresults(vector<double>& result,
      vector<double>& errormargin) const
   {
   const double cfactor = libbase::Qinv((1.0 - confidence) / 2.0);
   // determine a new estimate
   system->estimate(result, errormargin);
   assert(result.size() == errormargin.size());
   // determine confidence interval from standard error
   errormargin *= cfactor;
   }

/*!
 * \brief Initialize given slave
 * \param   s              Slave to be initialized
 * \param   systemstring   Serialized system description
 *
 * Initialize given slave by sending the system being simulated and the
 * current simulation parameter.
 */
void montecarlo::initslave(slave *s, std::string systemstring)
   {
   if (!call(s, "slave_getcode"))
      return;
   if (!send(s, systemstring))
      return;
   if (!call(s, "slave_getparameter"))
      return;
   if (!send(s, system->get_parameter()))
      return;
   trace << "DEBUG (estimate): Slave (" << s << ") initialized ok."
         << std::endl;
   }

/*!
 * \brief Initialize any new slaves
 * \param   systemstring   Serialized system description
 *
 * If there are any slaves in the NEW state, initialize them by sending the system
 * being simulated and the current simulation parameters.
 */
void montecarlo::initnewslaves(std::string systemstring)
   {
   while (slave *s = newslave())
      {
      trace << "DEBUG (estimate): New slave found (" << s << "), initializing."
            << std::endl;
      initslave(s, systemstring);
      }
   }

/*!
 * \brief Get idle slaves to work if we're not yet done
 * \param   converged  True if results have already converged
 *
 * If there are any slaves in the IDLE state, ask them to start working. We ask *all* IDLE
 * slaves to work, as long as the results have not yet converged. Therefore, this happens
 * when the target accuracy is not yet reached or if the number of samples gathered is not
 * yet enough. This necessarily causes extraneous results to be computed; these will then
 * be discarded during the next turn. This method avoids the master hanging up waiting for
 * results from slaves that will never come (happens if the machine is locked up but the
 * TCP/IP stack is still running).
 */
void montecarlo::workidleslaves(bool converged)
   {
   for (slave *s; (!converged) && (s = idleslave());)
      {
      trace << "DEBUG (estimate): Idle slave found (" << s
            << "), assigning work." << std::endl;
      if (!call(s, "slave_work"))
         continue;
      trace << "DEBUG (estimate): Slave (" << s << ") work assigned ok."
            << std::endl;
      }
   }

/*!
 * \brief Read and accumulate results from any pending slaves
 * \return  True if any new results have been added, false otherwise
 *
 * If there are any slaves in the EVENT_PENDING state, read their results. Values
 * returned are accumulated into the running totals.
 *
 * If any slave returns a result that does not correspond to the same system
 * or parameter that are now being simulated, this is discarded and the slave
 * is marked as 'new'.
 */
bool montecarlo::readpendingslaves()
   {
   bool results_available = false;
   while (slave *s = pendingslave())
      {
      trace << "DEBUG (estimate): Pending event from slave (" << s
            << "), trying to read." << std::endl;
      // get digest and parameter for simulated system
      std::string simdigest;
      double simparameter;
      if (!receive(s, simdigest) || !receive(s, simparameter))
         continue;
      // set up space for results that need to be returned
      libbase::int64u estsamplecount = 0;
      vector<double> eststate;
      // get results
      if (!receive(s, estsamplecount) || !receive(s, eststate))
         continue;
      // check that results correspond to system under simulation
      if (std::string(sysdigest) != simdigest || simparameter
            != system->get_parameter())
         {
         trace << "DEBUG (estimate): Slave returned invalid results (" << s
               << "), re-initializing." << std::endl;
         resetslave(s);
         continue;
         }
      // accumulate
      system->accumulate_state(estsamplecount, eststate);
      // update usage information and return flag
      updatecputime(s);
      results_available = true;
      trace << "DEBUG (estimate): Read from slave (" << s << ") succeeded."
            << std::endl;
      }
   return results_available;
   }

// Main process

/*!
 * \brief Simulate the system until convergence to given accuracy & confidence,
 * and return estimated results
 * \param[out] result      Vector of results
 * \param[out] errormargin Vector of corresponding margin of error (radius of confidence interval)
 */
void montecarlo::estimate(vector<double>& result, vector<double>& errormargin)
   {
   t.start();

   // Initialise running values
   system->reset();
   // create string representation of system
   std::string systemstring = get_systemstring();
   // compute its digest
   std::istringstream is(systemstring);
   sysdigest.process(is);

   // Initialize results-writing system (if we're using it)
   if (resultsfile::isinitialized())
      setupfile();

   // Set up for master-slave system (if necessary)
   // and seed the experiment
   if (isenabled())
      {
      resetslaves();
      resetcputime();
      }
   else
      seed_experiment();

   // Repeat the experiment until all the following are true:
   // 1) We have the accuracy we need
   // 2) We have enough samples for the accuracy to be meaningful
   // An interrupt from the user overrides everything...
   bool converged = false;
   while (!converged)
      {
      bool results_available = false;
      // repeat the experiment
      if (isenabled())
         {
         // first initialize any new slaves
         initnewslaves(systemstring);
         // get idle slaves to work if we're not yet done
         workidleslaves(converged);
         // wait for results, but not indefinitely - this allows user to break
         waitforevent(true, 0.5);
         // accumulate results from any pending slaves
         results_available = readpendingslaves();
         }
      else
         {
         sampleandaccumulate();
         results_available = true;
         }
      // if we did get any results, update the statistics
      if (results_available)
         {
         updateresults(result, errormargin);
         // if we have done enough samples, check accuracy reached
         if (system->get_samplecount() >= libbase::int64u(min_samples))
            {
            switch (mode)
               {
               case mode_relative_error:
                  {
                  // determine error margin as a fraction of result mean
                  const vector<double> result_acc = errormargin / result;
                  // check if this is less than threshold
                  if (result_acc.max() <= threshold)
                     converged = true;
                  break;
                  }
               case mode_absolute_error:
                  {
                  // check if error margin is less than threshold
                  if (errormargin.max() <= threshold)
                     converged = true;
                  break;
                  }
               case mode_accumulated_result:
                  {
                  // determine the absolute accumulated result
                  vector<double> result_acc = result;
                  for (int i = 0; i < result_acc.size(); i++)
                     result_acc(i) *= system->get_samplecount(i);
                  // check if this is more than threshold
                  if (result_acc.min() >= threshold)
                     converged = true;
                  break;
                  }
               default:
                  failwith("Convergence mode not supported.");
                  break;
               }
            }
         // print something to inform the user of our progress
         display(result, errormargin);
         // write interim results
         if (resultsfile::isinitialized())
            writeinterimresults(result, errormargin);
         }
      // consider our work done if the user has interrupted the processing
      // (note: this overrides everything)
      if (interrupt())
         break;
      }

   // write final results
   if (resultsfile::isinitialized())
      writefinalresults(result, errormargin, interrupt());

   t.stop();
   }

} // end namespace
