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

   \section conventions Coding Conventions and Formatting
   Conventions for code formatting that are used within this project include:
   - Use of spaces rather than tabs in text files; tabs, when present, should
     be replaced by spaces.
   - Indenting using 3-space, at all levels
   - Indenting braces, which are to be usually on separate lines
   - Wherever possible, lines should be kept to less than 80 characters width;
     please use line-breakers only where strictly necessary
   - Doxygen documentation should use '\' to indicate keywords; short-style
     comment forms should only be used where a brief description only is to
     be given.

   When adding to or modifying existing code, please keep to the coding style
   already present, as far as possible, unless this is already non-conforming.
   In view of this, please set your editor preferences accordingly, to avoid
   needless automatic editing, which complicates merges.
   Edits which are primarily sylistic should be committed separately.
*/
