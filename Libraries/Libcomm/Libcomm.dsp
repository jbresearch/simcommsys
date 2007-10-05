# Microsoft Developer Studio Project File - Name="LibComm" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=LibComm - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "LibComm.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "LibComm.mak" CFG="LibComm - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "LibComm - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "LibComm - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "LibComm - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 2
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /MD /W3 /WX /vd0 /GX /Zi /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /D "_AFXDLL" /FR /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG" /d "_AFXDLL"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "LibComm - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 2
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /WX /Gm /vd0 /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /D "_AFXDLL" /FR /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG" /d "_AFXDLL"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ENDIF 

# Begin Target

# Name "LibComm - Win32 Release"
# Name "LibComm - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\Source\anneal_interleaver.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\anneal_puncturing.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\anneal_system.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\annealer.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\awgn.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\bcjr.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\berrou.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\channel.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\codec.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\commsys.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\commsys_bitprofiler.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\commsys_profiler.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\crypt.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\diffturbo.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\dvbcrsc.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\experiment.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\file_lut.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\flat.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\fsm.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\gcc.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\helical.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\interleaver.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\lapgauss.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\laplacian.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\lut_interleaver.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\mapcc.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\md5.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\modulator.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\montecarlo.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\mpsk.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\named_lut.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\nrcc.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\onetimepad.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\padded.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\plmod.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\puncture.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\puncture_file.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\puncture_null.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\puncture_stipple.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\rand_lut.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\rc4.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\rectangular.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\rscc.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\serializer_libcomm.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\sha.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\shift_lut.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\sigspace.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\stream_lut.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\turbo.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\uncoded.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\uniform_lut.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\vale96int.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\Source\anneal_interleaver.h
# End Source File
# Begin Source File

SOURCE=.\Source\anneal_puncturing.h
# End Source File
# Begin Source File

SOURCE=.\Source\anneal_system.h
# End Source File
# Begin Source File

SOURCE=.\Source\annealer.h
# End Source File
# Begin Source File

SOURCE=.\Source\awgn.h
# End Source File
# Begin Source File

SOURCE=.\Source\bcjr.h
# End Source File
# Begin Source File

SOURCE=.\Source\berrou.h
# End Source File
# Begin Source File

SOURCE=.\Source\channel.h
# End Source File
# Begin Source File

SOURCE=.\Source\codec.h
# End Source File
# Begin Source File

SOURCE=.\Source\commsys.h
# End Source File
# Begin Source File

SOURCE=.\Source\commsys_bitprofiler.h
# End Source File
# Begin Source File

SOURCE=.\Source\commsys_profiler.h
# End Source File
# Begin Source File

SOURCE=.\Source\crypt.h
# End Source File
# Begin Source File

SOURCE=.\Source\diffturbo.h
# End Source File
# Begin Source File

SOURCE=.\Source\dvbcrsc.h
# End Source File
# Begin Source File

SOURCE=.\Source\experiment.h
# End Source File
# Begin Source File

SOURCE=.\Source\file_lut.h
# End Source File
# Begin Source File

SOURCE=.\Source\flat.h
# End Source File
# Begin Source File

SOURCE=.\Source\fsm.h
# End Source File
# Begin Source File

SOURCE=.\Source\gcc.h
# End Source File
# Begin Source File

SOURCE=.\Source\helical.h
# End Source File
# Begin Source File

SOURCE=.\Source\interleaver.h
# End Source File
# Begin Source File

SOURCE=.\Source\lapgauss.h
# End Source File
# Begin Source File

SOURCE=.\Source\laplacian.h
# End Source File
# Begin Source File

SOURCE=.\Source\lut_interleaver.h
# End Source File
# Begin Source File

SOURCE=.\Source\mapcc.h
# End Source File
# Begin Source File

SOURCE=.\Source\md5.h
# End Source File
# Begin Source File

SOURCE=.\Source\modulator.h
# End Source File
# Begin Source File

SOURCE=.\Source\montecarlo.h
# End Source File
# Begin Source File

SOURCE=.\Source\mpsk.h
# End Source File
# Begin Source File

SOURCE=.\Source\named_lut.h
# End Source File
# Begin Source File

SOURCE=.\Source\nrcc.h
# End Source File
# Begin Source File

SOURCE=.\Source\onetimepad.h
# End Source File
# Begin Source File

SOURCE=.\Source\padded.h
# End Source File
# Begin Source File

SOURCE=.\Source\plmod.h
# End Source File
# Begin Source File

SOURCE=.\Source\puncture.h
# End Source File
# Begin Source File

SOURCE=.\Source\puncture_file.h
# End Source File
# Begin Source File

SOURCE=.\Source\puncture_null.h
# End Source File
# Begin Source File

SOURCE=.\Source\puncture_stipple.h
# End Source File
# Begin Source File

SOURCE=.\Source\rand_lut.h
# End Source File
# Begin Source File

SOURCE=.\Source\rc4.h
# End Source File
# Begin Source File

SOURCE=.\Source\rectangular.h
# End Source File
# Begin Source File

SOURCE=.\Source\rscc.h
# End Source File
# Begin Source File

SOURCE=.\Source\serializer_libcomm.h
# End Source File
# Begin Source File

SOURCE=.\Source\sha.h
# End Source File
# Begin Source File

SOURCE=.\Source\shift_lut.h
# End Source File
# Begin Source File

SOURCE=.\Source\sigspace.h
# End Source File
# Begin Source File

SOURCE=.\Source\stream_lut.h
# End Source File
# Begin Source File

SOURCE=.\Source\turbo.h
# End Source File
# Begin Source File

SOURCE=.\Source\uncoded.h
# End Source File
# Begin Source File

SOURCE=.\Source\uniform_lut.h
# End Source File
# Begin Source File

SOURCE=.\Source\vale96int.h
# End Source File
# End Group
# End Target
# End Project
