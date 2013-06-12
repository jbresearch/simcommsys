Executables Tagging
===================

For Exectutables to be tagged properly with the branch label, you need to load the macro 
file SimCommSys.vsmacros.  Unfortunately this would apply to all Visual Studio solutions,
so you can only load one whatever branch you are in.  Therefore it is suggested that you load
the one in the trunk.  If you need to modify this in a particular branch, it is suggested
that you temporarily unload the macro file from trunk and load the one from the required
branch.

N.B. The macro file includes also a macro that stops the build process when a
a build error is encountered.  The normal behaviour of VIsual Studio is to proceed the
build process.  With a Solution with lots of projects this could be a problem.


Boost
=====

Since the Solution may be build either as 32-bit (Win32) or 64-bit (x64), you need two 
versions for the Boost libraries, one for each platform.  The location of the Boost root
directory should be pointed to by the environmental variable BOOST_ROOT.  The 32-bit 
libraries are assumed to be under BOOST_ROOT/lib/w32, wheras the 64-bit libraries are
assumed to be under BOOST_ROOT/lib/x64.

In order to build 64-bit version of Boost, you need to issue the command

	bjam --toolset=msvc-10.0 address-model=64 --build-type=complete stage

Then move the contents of stage\lib to lib\x64

For the 32-bit version, you need the command

	bjam --toolset=msvc-9.0 --build-type=complete stage

Then move the contents of stage\lib to lib\win32

Between builds remove the directories bin.v2 and stage.

Installation
============

To install the 32-bit binaries right click on the "Install Win32" project file under the
folder Installer and choose "Install".  If this is greyed out, you need to build the solution
first.  Make sure that you select the Win32 platform when doing so.  It is recommended that 
the binaries are installed under c:\Program Files (x86)\SimCommSys, which is the default 
option in this case.

To install the 64-bit binaries right click on the "Install x64" project file under the
folder Installer and choose "Install".  If this is greyed out, you need to build the solution
first.  Make sure that you select the x64 platform when doing so.  It is recommended that 
the binaries are installed under c:\Program Files\SimCommSys, which is the default 
option in this case.