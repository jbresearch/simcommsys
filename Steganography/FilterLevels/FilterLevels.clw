; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CDisplayResultsDlg
LastTemplate=CDialog
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "filterlevels.h"
LastPage=0

ClassCount=4
Class1=CFilterLevelsApp
Class2=CAboutDlg
Class3=CFilterLevelsDlg

ResourceCount=3
Resource1=IDD_ABOUTBOX
Resource2=IDD_DIALOG1
Class4=CDisplayResultsDlg
Resource3=IDD_DISPLAY_RESULTS

[CLS:CFilterLevelsApp]
Type=0
BaseClass=CWinApp
HeaderFile=FilterLevels.h
ImplementationFile=FilterLevels.cpp

[CLS:CAboutDlg]
Type=0
BaseClass=CDialog
HeaderFile=FilterLevelsDlg.h
ImplementationFile=FilterLevelsDlg.cpp
LastObject=65535

[CLS:CFilterLevelsDlg]
Type=0
BaseClass=CDialog
HeaderFile=FilterLevelsDlg.h
ImplementationFile=FilterLevelsDlg.cpp

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=3
Control1=65535,static,1342308480
Control2=65535,static,1342308352
Control3=IDOK,button,1342373889

[DLG:IDD_DIALOG1]
Type=1
Class=CFilterLevelsDlg
ControlCount=8
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_WEIGHT,edit,1350639744
Control4=IDC_STATIC,static,1342308352
Control5=IDC_KEEPNOISE,button,1342242819
Control6=IDC_SLIDER,msctls_trackbar32,1342242817
Control7=IDC_STATIC,static,1342308352
Control8=IDC_STATIC,static,1342308352

[DLG:IDD_DISPLAY_RESULTS]
Type=1
Class=CDisplayResultsDlg
ControlCount=6
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_STATIC,static,1342308352
Control5=IDC_WHITE,static,1342308864
Control6=IDC_BLACK,static,1342308864

[CLS:CDisplayResultsDlg]
Type=0
HeaderFile=DisplayResultsDlg.h
ImplementationFile=DisplayResultsDlg.cpp
BaseClass=CDialog
Filter=D
LastObject=IDC_BLACK
VirtualFilter=dWC

