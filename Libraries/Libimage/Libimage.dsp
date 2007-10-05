# Microsoft Developer Studio Project File - Name="LibImage" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=LibImage - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "LibImage.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "LibImage.mak" CFG="LibImage - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "LibImage - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "LibImage - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "LibImage - Win32 Release"

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

!ELSEIF  "$(CFG)" == "LibImage - Win32 Debug"

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

# Name "LibImage - Win32 Release"
# Name "LibImage - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\Source\atmfilter.cpp

!IF  "$(CFG)" == "LibImage - Win32 Release"

# ADD CPP /Ze

!ELSEIF  "$(CFG)" == "LibImage - Win32 Debug"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\Source\awfilter.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\filter.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\limiter.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\picture.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\pixel.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\variancefilter.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\wavelet.cpp
# End Source File
# Begin Source File

SOURCE=.\Source\waveletfilter.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\Source\atmfilter.h
# End Source File
# Begin Source File

SOURCE=.\Source\awfilter.h
# End Source File
# Begin Source File

SOURCE=.\Source\filter.h
# End Source File
# Begin Source File

SOURCE=.\Source\limiter.h
# End Source File
# Begin Source File

SOURCE=.\Source\picture.h
# End Source File
# Begin Source File

SOURCE=.\Source\pixel.h
# End Source File
# Begin Source File

SOURCE=.\Source\variancefilter.h
# End Source File
# Begin Source File

SOURCE=.\Source\wavelet.h
# End Source File
# Begin Source File

SOURCE=.\Source\waveletfilter.h
# End Source File
# End Group
# End Target
# End Project
