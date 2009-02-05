/*!
   \mainpage SimCommSys Project
   \author   Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \section intro Introduction
   This project consists of a set of libraries for simulating communication
   systems. Of principal interest is the error-control coding component,
   where various kinds of binary and non-binary turbo codecs are implemented.
   Also in this project are a number of targets implementing various stages
   of the communication system (such as the encoder, decoder, and a complete
   simulator).

   \section classes Principal Classes
   The most important classes in this project are:
   - libcomm::montecarlo, which implements a distributed Monte Carlo simulator.
     The simulator automatically detects convergence and terminates according
     to the required confidence and tolerance level.
   - libcomm::commsys_simulator, which implements a libcomm::experiment as
     required by libcomm::montecarlo for simulating the performance of a
     complete communication system.
   - libcomm::commsys, which defines a complete block-based communication
     system, including:
     - libcomm::codec, the error-control codec
     - libcomm::mapper, a symbol-mapping between the codec and modulator
     - libcomm::blockmodem, the modulation scheme
     - libcomm::channel, a model for the communications channel
*/
