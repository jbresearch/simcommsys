#ifndef __montecarlo_h
#define __montecarlo_h

#include "config.h"

#include "timer.h"
#include "sha.h"
#include "experiment.h"
#include "masterslave.h"

namespace libcomm {

/*!
   \brief   Monte Carlo Estimator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (2 Sep 1999)
   added a hook for clients to know the number of frames simulated in a particular run.

   \version 1.11 (9 Sep 1999)
   changed min_passes to 128, since we observed that with 30 samples only, the distribution
   is still quite skewed.

   \version 1.12 (26 Oct 2001)
   added a virtual display function, to facilitate deriving from the class to produce a
   windowed GUI (by using custom display routines), and also added a virtual interrupt
   function to allow a derived class to stop the processing routine. Both functions are
   protected so they can only be called by the class itself or by derived classes.

   \version 1.13 (26 Oct 2001)
   added an empty creator object which doesn't initialise the simulator and doesn't bind
   it to a system - this is necessary for deriving classes from this. Also created separate
   initialisation and finalisation routines to allow re-use of the same montecarlo object.

   \version 1.14 (16 Nov 2001)
   added a virtual destructor.

   \version 1.15 (23 Feb 2002)
   added flushes to all end-of-line clog outputs, to clean up text user interface.

   \version 1.16 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.
   also changed use of iostream from global to std namespace.

   \version 1.17 (25 Mar 2002)
   changed min_passes to min_samples, to indicate that this is the number of minimum
   samples, not of minimum cycles, that are necessary to ensure a gaussian distribution.
   Note that in each pass, there will really be a number of samples, depending on how
   long each sample takes.

   \version 1.18 (27 Mar 2002)
   removed the timer display on clog for the estimate function.

   \version 1.20 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.30 (20 Apr 2007)
   - updated to use masterslave instead of cmpi class, to support socket-based model
   - names of private functions changed from child to slave, to reflect the model better

   \version 1.31 (24-25 Apr 2007)
   - added a timer element, used by the default display() routine, and also
    provided a public interface for use by clients.
   - updated slave initialization so that we also receive the code itself
   - refactored by splitting the slave init into two routines, one for the code
    and one for the SNR; this facilitates multiple simulations on the same code
   - updated to conform with the changes in masterslave 1.10, where the slave
    functions are no longer static, and the class is meant to be instantiated
    by derivation

   \version 1.32 (20 Nov 2007)
   - added timeout when waiting for an event from the slaves in estimate()
   - modified control structure in estimate() by removing the infinite loop when
    parallel execution is enabled; this means that each run of the outer loop does
    not necessarily mean that a new result set is available. This condition is
   controlled by a new variable within the loop.
   - modified ending-logic, so that:
    - all end conditions are clearly specified in the loop condition
    - a user interrupt now overrides everything, we don't even wait for slaves
      to finish any pending work
   - modified slave-handling logic, so that new work only gets posted if the sum
    of samples already aggregated and the number of pending slaves is less than
    the minimum threshold; this assumes that only one sample is returned per slave,
    and allows an earlier exit during fast result convergence in large clusters.
    Clearly this only applies when the actual accuracy has already been reached.

   \version 1.33 (26 Nov 2007)
   - optimized display routine, so that updates are rate-limited

   \version 1.34 (30 Nov 2007)
   - added error checking on assigning work to slaves

   \version 1.35 (17 Dec 2007)
   - added printing of seed when generated

   \version 1.40 (18 Dec 2007)
   - Updated according to the new definition of experiment::sample(), which only
     returns a single pass.
   - Extracted result accumulation to updateresults()
   - Extracted initialization of new slaves
   - Extracted getting idle slaves to work
   - Moved accumulation of results to a new function, separating the accumulation
     process from the actual updating of final results; modified updateresults()
     accordingly
   - Modified updateresults() so that it is no longer given the number of passes,
     but instead uses the (already updated) samplecount; also made this function
     const, as it should not be updating any class members.
   - Extracted reading and accumulation of results from pending slaves.
   - Reads as many results as are pending, not merely the first one.
   - Removed bail-out facility
   - Extracted computing a single sample and accumulating result
   - Added handling of minimum granularity in slave_work; this requires the slaves
     to perform result accumulation. Therefore slaves no longer return just a single
     estimate vector, but rather the sample count and also vectors with sample sums,
     and sums of squares.

   \version 1.41 (20 Dec 2007)
   - Fixed memory leak where system was not deleted in slaves.
   - Made getters const members

   \version 1.42 (4 Jan 2008)
   - Modified accumulateresults() to use vector apply() rather than multiplication,
     when squaring the estimates. This change became necessary when vector mul was
     made private to avoid ambiguity.

   \version 1.43 (16 Jan 2008)
   - Minor change in estimate() to show the smallest, rather than the first, result
     when doing the real-time display. This is more indicative of the performance
     of the system at this point.
   - Modified slave_work() to compute its own result set and display progress

   \version 1.44 (21 Jan 2008)
   - Modified so that no-error-event accuracy is held as the largest positive double
     value, rather than zero; this makes it easier to distinguish from a genuine
     complete convergence, where the simulator now stops earlier.

   \version 1.45 (25 Jan 2008)
   - Added reset of cpu-time accumulation for slaves when starting a new estimate;
     this corrects the error in computing speedup on master.
   - Added tracking of system under simulation by keeping a digest of its string
     description.
   - Including system digest and current parameter with result set; this allows the
     master to discard any invalid results.
   - Extracted code to initialize a slave into a new function

   \version 1.46 (30 Jan 2008)
   - Fixed bug where initialise() was also setting tolerance limits
   - Removed constructor that also initializes system
   - Renamed finalise() to reset()

   \version 1.47 (1 Feb 2008)
   - Modified slave work-request behaviour: we now ask _all_ IDLE slaves to start
     working if the results have not yet converged.
   - Minor refactoring
   - Added debug information when changing accuracy and tolerance levels
*/

class montecarlo : public libbase::masterslave {
   /*! \name Object-wide constants */
   static const int  min_samples;   //!< minimum number of samples to assume gaussian distribution
   // @}
   /*! \name Bound objects */
   /*! \note If 'init' is false, and 'system' is not NULL, then there is a dynamically allocated
             object at this address. This should be deleted when no longer necessary.
   */
   bool           init;          //!< Flag to indicate that a system has been bound (only in master)
   experiment     *system;       //!< System being sampled            
   // @}
   /*! \name Internal variables */
   double         cfactor;       //!< factor dependent on confidence level
   double         accuracy;      //!< accuracy level required
   libbase::timer t;             //!< timer to keep track of running estimate
   sha            sysdigest;     //!< digest of the currently-simulated system
   // @}
   /*! \name Slave process functions & their functors */
   void slave_getcode(void);
   void slave_getsnr(void);
   void slave_work(void);
   libbase::specificfunctor<montecarlo> *fgetcode;
   libbase::specificfunctor<montecarlo> *fgetsnr;
   libbase::specificfunctor<montecarlo> *fwork;
   // @}
private:
   /*! \name Helper functions */
   void createfunctors(void);
   void destroyfunctors(void);
   // @}
   /*! \name Main estimator helper functions */
   void sampleandaccumulate();
   void updateresults(libbase::vector<double>& result, libbase::vector<double>& tolerance) const;
   void initslave(slave *s, std::string systemstring);
   void initnewslaves(std::string systemstring);
   void workidleslaves(bool converged);
   bool readpendingslaves();
protected:
   // @}
   /*! \name Overrideable user-interface functions */
   virtual bool interrupt() { return false; };
   virtual void display(const int pass, const double cur_accuracy, const double cur_mean);
   // @}
public:
   /*! \name Constructor/destructor */
   montecarlo();
   virtual ~montecarlo();
   // @}
   /*! \name Simulation initialization/finalization */
   void initialise(experiment *system);
   void reset();
   // @}
   /*! \name Simulation parameters */
   void set_confidence(const double confidence);   //!< Set confidence limit, say, 0.95 => 95% probability
   void set_accuracy(const double accuracy);       //!< Set target accuracy, say, 0.10 => 10% of mean
   // @}
   /*! \name Simulation results */
   //! Number of samples taken to produce the result
   int get_samplecount() const { return system->get_samplecount(); };
   //! Time taken to produce the result
   const libbase::timer& get_timer() const { return t; };
   // @}
   /*! \name Main process */
   void estimate(libbase::vector<double>& result, libbase::vector<double>& tolerance);
   // @}
};

}; // end namespace

#endif
