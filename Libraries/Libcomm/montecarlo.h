#ifndef __montecarlo_h
#define __montecarlo_h

#include "config.h"

#include "timer.h"
#include "sha.h"
#include "experiment.h"
#include "masterslave.h"
#include "resultsfile.h"

namespace libcomm {

/*!
 * \brief   Monte Carlo Estimator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class montecarlo : public libbase::masterslave, private resultsfile {
   /*! \name Object-wide constants */
   static const libbase::int64u min_samples; //!< minimum number of samples
   // @}
   /*! \name Bound objects */
   /*! \note If 'init' is false, and 'system' is not NULL, then there is a dynamically allocated
    * object at this address. This should be deleted when no longer necessary.
    */
   bool bound; //!< Flag to indicate that a system has been bound (only in master)
   experiment *system; //!< System being sampled
   // @}
   /*! \name Internal variables */
   double confidence; //!< confidence level required
   double accuracy; //!< accuracy level required
   libbase::timer t; //!< timer to keep track of running estimate
   sha sysdigest; //!< digest of the currently-simulated system
   // @}
   /*! \name Slave process functions & their functors */
   void slave_getcode(void);
   void slave_getparameter(void);
   void slave_work(void);
   libbase::specificfunctor<montecarlo> *fgetcode;
   libbase::specificfunctor<montecarlo> *fgetparameter;
   libbase::specificfunctor<montecarlo> *fwork;
   // @}
private:
   /*! \name Helper functions */
   std::string get_systemstring();
   void seed_experiment();
   void createfunctors(void);
   void destroyfunctors(void);
   // @}
   /*! \name Main estimator helper functions */
   void sampleandaccumulate();
   void updateresults(libbase::vector<double>& result,
         libbase::vector<double>& tolerance) const;
   void initslave(slave *s, std::string systemstring);
   void initnewslaves(std::string systemstring);
   void workidleslaves(bool converged);
   bool readpendingslaves();
   // @}
protected:
   // System-specific file-handler functions
   void writeheader(std::ostream& sout) const;
   void writeresults(std::ostream& sout, libbase::vector<double>& result,
         libbase::vector<double>& tolerance) const;
   void writestate(std::ostream& sout) const;
   void lookforstate(std::istream& sin);
   /*! \name Overrideable user-interface functions */
   /*! \brief User-interrupt check
    * This function should return true if the user has requested an interrupt.
    * Once it returns true, all subsequent evaluations should keep returning
    * true again. Default action is to check for user pressing 'q' or Ctrl-C.
    */
   virtual bool interrupt()
      {
      static bool interrupted = false;
      if (interrupted)
         return true;
      if (libbase::interrupted())
         interrupted = true;
      else if (libbase::keypressed() > 0)
         interrupted = (libbase::readkey() == 'q');
      return interrupted;
      }
   virtual void display() const;
   // @}
public:
   /*! \name Constructor/destructor */
   montecarlo();
   virtual ~montecarlo();
   // @}
   /*! \name Simulation binding/releasing */
   void bind(experiment *system);
   void release();
   // @}
   /*! \name Simulation parameters */
   //! Set confidence limit, say, 0.95 => 95% probability
   void set_confidence(double confidence);
   //! Set target accuracy, say, 0.10 => 10% of mean
   void set_accuracy(double accuracy);
   //! Associates with given results file
   void set_resultsfile(const std::string& fname)
      {
      resultsfile::init(fname);
      }
   //! Get confidence limit
   double get_confidence() const
      {
      return confidence;
      }
   //! Get target accuracy
   double get_accuracy() const
      {
      return accuracy;
      }
   // @}
   /*! \name Simulation results */
   //! Number of samples taken to produce the result
   libbase::int64u get_samplecount() const
      {
      return system->get_samplecount();
      }
   //! Time taken to produce the result
   const libbase::timer& get_timer() const
      {
      return t;
      }
   // @}
   /*! \name Main process */
   void estimate(libbase::vector<double>& result,
         libbase::vector<double>& tolerance);
   // @}
};

} // end namespace

#endif
