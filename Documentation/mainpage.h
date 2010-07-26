/*!
 * \mainpage SimCommSys Project
 * \author   Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \section intro Introduction
 * The SimCommSys project consists of a set of libraries for simulating
 * communication systems using a distributed Monte Carlo simulator. Of
 * principal interest is the error-control coding component, where various
 * kinds of binary and non-binary codes are implemented, including turbo,
 * LDPC, repeat-accumulate, and Reed-Solomon.
 * This project also contains a number of executables implementing various
 * stages of the communication system (such as the encoder, decoder, and a
 * complete simulator) and a system benchmark.
 *
 * \section classes Principal Classes
 * The most important classes in this project are:
 * - libcomm::montecarlo, which implements a distributed Monte Carlo simulator.
 * The simulator automatically detects convergence and terminates according
 * to the required confidence and tolerance level.
 * - libcomm::commsys_simulator, which implements a libcomm::experiment as
 * required by libcomm::montecarlo for simulating the performance of a
 * complete communication system.
 * - libcomm::commsys, which defines a complete block-based communication
 * system, including:
 * - libcomm::codec, the error-control codec
 * - libcomm::mapper, a symbol-mapping between the codec and modulator
 * - libcomm::blockmodem, the modulation scheme
 * - libcomm::channel, a model for the communications channel
 *
 * \section conventions Coding Conventions and Formatting
 * Conventions for code formatting that are used within this project include:
 * - Use of spaces rather than tabs in text files; tabs, when present, should
 * be replaced by spaces.
 * - Indenting using 3-space, at all levels
 * - Indenting braces, which are to be usually on separate lines
 * - Wherever possible, lines should be kept to less than 80 characters width;
 * please use line-breakers only where strictly necessary
 * - Doxygen documentation should use '\' to indicate keywords; short-style
 * comment forms should only be used where a brief description only is to
 * be given.
 *
 * When adding to or modifying existing code, please keep to the coding style
 * already present, as far as possible, unless this is already non-conforming.
 * In view of this, please set your editor preferences accordingly, to avoid
 * needless automatic editing, which complicates merges.
 * Edits which are primarily sylistic should be committed separately.
 *
 * \section software Software Required
 * The code base is meant to be edited using an IDE; requirements depend on
 * the operating system:
 * - Windows development assumes Microsoft Visual Studio 2005; the Team
 * Edition is suggested for its performance-related tools. The Refactor!
 * plugin from Developer Express Inc. is also recommended.
 * - Linux development assumes Eclipse 3.5, with Subclipse 1.4.x for SVN
 * integration (install both the main items and the SVNKit to work in
 * Ubuntu 9.04) and Eclox for Doxygen access.
 *
 * Compilation on Windows uses the Microsoft built-in compiler; on Linux G++
 * is used. Any recent G++ package should work - the current reference
 * standard is 4.3.3 as in Ubuntu 9.04. Both operating systems need the
 * development version of Boost v1.35 installed, together with the 'Program
 * Options' optional component. For Windows, install the Mutithread and
 * Mutithread Debug variants.
 */
