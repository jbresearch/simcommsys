# Microsoft Developer Studio Project File - Name="LibBase" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=LibBase - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "LibBase.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "LibBase.mak" CFG="LibBase - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "LibBase - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "LibBase - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "LibBase - Win32 Release"

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

!ELSEIF  "$(CFG)" == "LibBase - Win32 Debug"

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

# Name "LibBase - Win32 Release"
# Name "LibBase - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\Source\bitfield.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\bstream.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\cmpi.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\config.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\fastsecant.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\fbstream.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\histogram.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\itfunc.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\logreal.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\logrealfast.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\matrix.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\matrix3.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\mpgnu.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\mpreal.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\randgen.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\rvstatistics.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\secant.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\serializer.cpp
# ADD CPP /Ze
# End Source File
# Begin Source File

SOURCE=.\Source\timer.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\vcs.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\vector.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\Source\bitfield.h
# End Source File
# Begin Source File

SOURCE=.\Source\bstream.h
# End Source File
# Begin Source File

SOURCE=.\Source\cmpi.h
# End Source File
# Begin Source File

SOURCE=.\Source\config.h
# End Source File
# Begin Source File

SOURCE=.\Source\fastsecant.h
# End Source File
# Begin Source File

SOURCE=.\Source\fbstream.h
# End Source File
# Begin Source File

SOURCE=.\Source\histogram.h
# End Source File
# Begin Source File

SOURCE=.\Source\itfunc.h
# End Source File
# Begin Source File

SOURCE=.\Source\logreal.h
# End Source File
# Begin Source File

SOURCE=.\Source\logrealfast.h
# End Source File
# Begin Source File

SOURCE=.\Source\matrix.h
# End Source File
# Begin Source File

SOURCE=.\Source\matrix3.h
# End Source File
# Begin Source File

SOURCE=.\Source\mpgnu.h
# End Source File
# Begin Source File

SOURCE=.\Source\mpreal.h
# End Source File
# Begin Source File

SOURCE=.\Source\randgen.h
# End Source File
# Begin Source File

SOURCE=.\Source\rvstatistics.h
# End Source File
# Begin Source File

SOURCE=.\Source\secant.h
# End Source File
# Begin Source File

SOURCE=.\Source\serializer.h
# End Source File
# Begin Source File

SOURCE=.\Source\timer.h
# End Source File
# Begin Source File

SOURCE=.\Source\vcs.h
# End Source File
# Begin Source File

SOURCE=.\Source\vector.h
# End Source File
# End Group
# End Target
# End Project
