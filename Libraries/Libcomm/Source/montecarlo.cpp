/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "montecarlo.h"

#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "truerand.h"
#include "randgen.h"
#include <sstream>
#include <limits>

namespace libcomm {

using std::cerr;
using libbase::trace;
using libbase::vector;

const libbase::int64u montecarlo::min_samples = 128;

// worker processes

void montecarlo::slave_getcode(void)
   {
   delete system;
   // Receive system as a string
   std::string systemstring;
   if(!receive(systemstring))
      exit(1);
   // Create system object from serialization
   std::istringstream is(systemstring);
   is >> system;
   // Compute its digest
   is.seekg(0);
   sysdigest.process(is);
   // Tell the user what we've done
   cerr << "Date: " << libbase::timer::date() << "\n";
   cerr << system->description() << "\n";
   cerr << "Digest: " << std::string(sysdigest) << "\n";
   }

void montecarlo::slave_getparameter(void)
   {
   cerr << "Date: " << libbase::timer::date() << "\n";

   seed_experiment();
   double x;
   if(!receive(x))
      exit(1);
   system->set_parameter(x);

   cerr << "Simulating system at parameter = " << system->get_parameter() << "\n";
   }

void montecarlo::slave_work(void)
   {
   // Initialise running values
   system->reset();

   // Iterate for 500ms, which is a good compromise between efficiency and usability
   libbase::timer t;
   while(t.elapsed() < 0.5)
      sampleandaccumulate();
   t.stop();   // to avoid expiry

   // Send system digest and current parameter back to master
   assertalways(send(sysdigest));
   assertalways(send(system->get_parameter()));
   // Send accumulated results back to master
   libbase::vector<double> state;
   system->get_state(state);
   assertalways(send(system->get_samplecount()));
   assertalways(send(state));

   // print something to inform the user of our progress
   vector<double> result, tolerance;
   updateresults(result, tolerance);
   const double acc = tolerance.max();
   display(system->get_samplecount(), (acc<1 ? 100*acc : 99), result.min());
   }

// helper functions

std::string montecarlo::get_systemstring()
   {
   std::ostringstream os;
   os << system;
   std::string systemstring = os.str();
   return systemstring;
   }

void montecarlo::seed_experiment()
   {
   libbase::truerand trng;
   libbase::randgen prng;
   const libbase::int32u seed = trng.ival();
   prng.seed(seed);
   system->seedfrom(prng);
   cerr << "Seed: " << seed << "\n";
   }

void montecarlo::createfunctors(void)
   {
   fgetcode = new libbase::specificfunctor<montecarlo>(this, &libcomm::montecarlo::slave_getcode);
   fgetparameter = new libbase::specificfunctor<montecarlo>(this, &libcomm::montecarlo::slave_getparameter);
   fwork = new libbase::specificfunctor<montecarlo>(this, &libcomm::montecarlo::slave_work);
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
   trace << "DEBUG (montecarlo): writing results header.\n";
   // Print information on the simulation being performed
   trace << "DEBUG (montecarlo): position before = " << sout.tellp() << "\n";
   sout << "#% " << system->description() << "\n";
   sout << "#% Tolerance: " << 100*accuracy << "%\n";
   sout << "#% Confidence: " << 100*confidence << "%\n";
   sout << "#% Date: " << libbase::timer::date() << "\n";
   sout << "#% URL: " << __WCURL__ << "\n";
   sout << "#% Version: " << __WCVER__ << "\n";
   sout << "#\n";
   // Print results header
   sout << "# Par";
   for(int i=0; i<system->count(); i++)
      sout << "\t" << system->result_description(i) << "\tTol";
   sout << "\tSamples\tCPUtime\n";
   sout << std::flush;
   trace << "DEBUG (montecarlo): position after = " << sout.tellp() << "\n";
   }

void montecarlo::writeresults(std::ostream& sout, libbase::vector<double>& result, libbase::vector<double>& tolerance) const
   {
   assert(sout.good());
   if(get_samplecount() == 0)
      return;
   trace << "DEBUG (montecarlo): writing results.\n";
   // Write current estimates to file
   trace << "DEBUG (montecarlo): position before = " << sout.tellp() << "\n";
   sout << system->get_parameter();
   for(int i=0; i<system->count(); i++)
      sout << '\t' << result(i) << '\t' << result(i)*tolerance(i);
   sout << '\t' << get_samplecount();
   sout << '\t' << getcputime() << '\n';
   sout << std::flush;
   trace << "DEBUG (montecarlo): position after = " << sout.tellp() << "\n";
   }

void montecarlo::writestate(std::ostream& sout) const
   {
   assert(sout.good());
   if(get_samplecount() == 0)
      return;
   trace << "DEBUG (montecarlo): writing state.\n";
   // Write accumulated values to file
   trace << "DEBUG (montecarlo): position before = " << sout.tellp() << "\n";
   libbase::vector<double> state;
   system->get_state(state);
   sout << "## System: " << sysdigest << '\n';
   sout << "## Parameter: " << system->get_parameter() << '\n';
   sout << "## Samples: " << get_samplecount() << '\n';
   sout << "## State: " << state.size();
   for(int i=0; i<state.size(); i++)
      sout << '\t' << state(i);
   sout << '\n';
   sout << std::flush;
   trace << "DEBUG (montecarlo): position after = " << sout.tellp() << "\n";
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
   trace << "DEBUG (montecarlo): looking for state.\n";
   sin.seekg(0);
   while(!sin.eof())
      {
      std::string s;
      getline(sin, s);
      if(s.substr(0,10) == "## System:")
         digest = s.substr(10);
      else if(s.substr(0,13) == "## Parameter:")
         std::istringstream(s.substr(13)) >> parameter;
      else if(s.substr(0,11) == "## Samples:")
         std::istringstream(s.substr(11)) >> samplecount;
      else if(s.substr(0,9) == "## State:")
         {
         std::istringstream is(s.substr(9));
         int count;
         is >> count;
         state.init(count);
         for(int i=0; i<count; i++)
            is >> state(i);
         }
      }
   // reset file
   sin.clear();
   // check that results correspond to system under simulation
   if(digest == std::string(sysdigest) && parameter == system->get_parameter())
      {
      cerr << "NOTICE: Reloading state with " << samplecount << " samples.\n";
      system->accumulate_state(samplecount, state);      
      }
   }

// overrideable user-interface functions

/*!
   \brief Default progress display routine.

   \note Display updates are rate-limited
*/
void montecarlo::display(libbase::int64u pass, double cur_accuracy, double cur_mean)
   {
   static libbase::timer tupdate;
   if(tupdate.elapsed() > 0.5)
      {
      using std::clog;
      const int prec = clog.precision(3);
      clog << "Timer: " << t << ", ";
      if(isenabled())
         clog << getnumslaves() << " clients, " << getcputime()/t.elapsed() << "x speedup, ";
      else
         clog << "local, " << t.cputime()/t.elapsed() << "x usage, ";
      clog << "pass " << pass << ", "
         << "[" << cur_mean << " +/- " << cur_accuracy << "%] \r" << std::flush;
      clog.precision(prec);
      tupdate.start();
      }
   }

// constructor/destructor

montecarlo::montecarlo()
   {
   createfunctors();
   bound = false;
   system = NULL;
   // set default parameter settings
   set_confidence(0.95);
   set_accuracy(0.10);
   }

montecarlo::~montecarlo()
   {
   release();
   delete system;
   destroyfunctors();
   }

// simulation initialization/finalization

void montecarlo::bind(experiment *system)
   {
   release();
   assert(montecarlo::system == NULL);
   bound = true;
   montecarlo::system = system;
   }

void montecarlo::release()
   {
   if(!bound)
      return;
   assert(system != NULL);
   bound = false;
   system = NULL;
   }

// simulation parameters

void montecarlo::set_confidence(double confidence)
   {
   assertalways(confidence > 0.5 && confidence < 1.0);
   trace << "DEBUG (montecarlo): setting confidence level of " << confidence << "\n";
   montecarlo::confidence = confidence;
   }

void montecarlo::set_accuracy(double accuracy)
   {
   assertalways(accuracy > 0 && accuracy < 1.0);
   trace << "DEBUG (montecarlo): setting accuracy level of " << accuracy << "\n";
   montecarlo::accuracy = accuracy;
   }

// main process

/*!
   \brief Compute a single sample and accumulate results
*/
void montecarlo::sampleandaccumulate()
   {
   vector<double> result;
   system->sample(result);
   system->accumulate(result);
   }

/*!
   \brief Determine overall estimate from accumulated results
   \param[out] result      Vector containing the set of estimates
   \param[out] tolerance   Corresponding confidence interval as a fraction of estimate
   
   \note If the accuracy cannot be computed yet (there has been no error event), then the
         accuracy reached takes the special largest-double value.
*/
void montecarlo::updateresults(vector<double>& result, vector<double>& tolerance) const
   {
   // init Qinv as the inverse of Q(), using the secant method
   libbase::secant Qinv(libbase::Q);
   const double cfactor = Qinv((1.0-confidence)/2.0);
   // determine a new estimate
   system->estimate(result, tolerance);
   assert(result.size() == tolerance.size());
   // determine confidence interval from standard error
   for(int i=0; i<result.size(); i++)
      {
      if(result(i) > 0)
         tolerance(i) *= cfactor/result(i);
      else
         tolerance(i) = std::numeric_limits<double>::max();
      }
   }

/*!
   \brief Initialize given slave
   \param   s              Slave to be initialized
   \param   systemstring   Serialized system description

   Initialize given slave by sending the system being simulated and the
   current simulation parameter.
*/
void montecarlo::initslave(slave *s, std::string systemstring)
   {
   if(!call(s, "slave_getcode"))
      return;
   if(!send(s, systemstring))
      return;
   if(!call(s, "slave_getparameter"))
      return;
   if(!send(s, system->get_parameter()))
      return;
   trace << "DEBUG (estimate): Slave (" << s << ") initialized ok.\n";
   }

/*!
   \brief Initialize any new slaves
   \param   systemstring   Serialized system description

   If there are any slaves in the NEW state, initialize them by sending the system
   being simulated and the current simulation parameters.
*/
void montecarlo::initnewslaves(std::string systemstring)
   {
   while(slave *s = newslave())
      {
      trace << "DEBUG (estimate): New slave found (" << s << "), initializing.\n";
      initslave(s, systemstring);
      }
   }

/*!
   \brief Get idle slaves to work if we're not yet done
   \param   converged  True if results have already converged

   If there are any slaves in the IDLE state, ask them to start working. We ask *all* IDLE
   slaves to work, as long as the results have not yet converged. Therefore, this happens
   when the target accuracy is not yet reached or if the number of samples gathered is not
   yet enough. This necessarily causes extraneous results to be computed; these will then
   be discarded during the next turn. This method avoids the master hanging up waiting for
   results from slaves that will never come (happens if the machine is locked up but the
   TCP/IP stack is still running).
*/
void montecarlo::workidleslaves(bool converged)
   {
   for(slave *s; (!converged) && (s = idleslave()); )
      {
      trace << "DEBUG (estimate): Idle slave found (" << s << "), assigning work.\n";
      if(!call(s, "slave_work"))
         continue;
      trace << "DEBUG (estimate): Slave (" << s << ") work assigned ok.\n";
      }
   }

/*!
   \brief Read and accumulate results from any pending slaves
   \return  True if any new results have been added, false otherwise

   If there are any slaves in the EVENT_PENDING state, read their results. Values
   returned are accumulated into the running totals.

   If any slave returns a result that does not correspond to the same system
   or parameter that are now being simulated, this is discarded and the slave
   is marked as 'new'.
*/
bool montecarlo::readpendingslaves()
   {
   bool results_available = false;
   while(slave *s = pendingslave())
      {
      trace << "DEBUG (estimate): Pending event from slave (" << s << "), trying to read.\n";
      // get digest and parameter for simulated system
      std::string simdigest;
      double simparameter;
      if(!receive(s, simdigest) || !receive(s, simparameter))
         continue;
      // set up space for results that need to be returned
      libbase::int64u estsamplecount = 0;
      vector<double> eststate;
      // get results
      if(!receive(s, estsamplecount) || !receive(s, eststate))
         continue;
      // check that results correspond to system under simulation
      if(std::string(sysdigest) != simdigest || simparameter != system->get_parameter())
         {
         trace << "DEBUG (estimate): Slave returned invalid results (" << s << "), re-initializing.\n";
         resetslave(s);
         continue;
         }
      // accumulate
      system->accumulate_state(estsamplecount, eststate);
      // update usage information and return flag
      updatecputime(s);
      results_available = true;
      trace << "DEBUG (estimate): Read from slave (" << s << ") succeeded.\n";
      }
   return results_available;
   }

// Main process

/*!
   \brief Simulate the system until convergence to given accuracy & confidence,
          and return estimated results
   \param[out] result      Vector of results
   \param[out] tolerance   Vector of corresponding result accuracy (at given confidence level)
*/
void montecarlo::estimate(vector<double>& result, vector<double>& tolerance)
   {
   t.start();

   // Initialise running values
   system->reset();
   // create string representation of system
   std::string systemstring = get_systemstring();
   // compute its digest
   std::istringstream is(systemstring);
   sysdigest.process(is);

   // Initialize results-writing system
   if(resultsfile::isinitialized())
      setupfile();

   // Set up for master-slave system (if necessary)
   // and seed the experiment
   if(isenabled())
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
   while(!converged)
      {
      bool results_available = false;
      // repeat the experiment
      if(isenabled())
         {
         // first initialize any new slaves
         initnewslaves(systemstring);
         // get idle slaves to work if we're not yet done
         workidleslaves(converged);
         // wait for results, but not indefinitely - this allows user to break
         trace << "DEBUG (estimate): Waiting for event.\n";
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
      if(results_available)
         {
         updateresults(result, tolerance);
         const double acc = tolerance.max();
         // check if we have reached the required accuracy
         if(acc <= accuracy && system->get_samplecount() >= min_samples)
            converged = true;
         // print something to inform the user of our progress
         display(system->get_samplecount(), (acc<1 ? 100*acc : 99), result.min());
         // write interim results
         if(resultsfile::isinitialized())
            writeinterimresults(result, tolerance);
         }
      // consider our work done if the user has interrupted the processing
      // (note: this overrides everything)
      if(interrupt())
         break;
      }

   // write final results
   if(resultsfile::isinitialized())
      writefinalresults(result, tolerance, interrupt());

   t.stop();
   }

}; // end namespace
