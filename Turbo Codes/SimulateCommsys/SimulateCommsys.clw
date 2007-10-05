; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CBenchmarkDlg
LastTemplate=CDialog
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "SimulateCommsys.h"
LastPage=0

ClassCount=10
Class1=CSimulateCommsysApp
Class2=CSimulateCommsysDoc
Class3=CSimulateCommsysView
Class4=CMainFrame

ResourceCount=7
Resource1=IDD_RANGE
Resource2=IDD_ABOUTBOX
Class5=CAboutDlg
Class6=CSelectChannelDlg
Resource3=IDD_ACCURACY
Class7=CSelectModulatorDlg
Resource4=IDD_CHANNEL
Resource5=IDD_MODULATOR
Class8=CSelectRangeDlg
Class9=CSelectAccuracyDlg
Resource6=IDD_BENCHMARK
Class10=CBenchmarkDlg
Resource7=IDR_MAINFRAME

[CLS:CSimulateCommsysApp]
Type=0
HeaderFile=SimulateCommsys.h
ImplementationFile=SimulateCommsys.cpp
Filter=N

[CLS:CSimulateCommsysDoc]
Type=0
HeaderFile=SimulateCommsysDoc.h
ImplementationFile=SimulateCommsysDoc.cpp
Filter=N

[CLS:CSimulateCommsysView]
Type=0
HeaderFile=SimulateCommsysView.h
ImplementationFile=SimulateCommsysView.cpp
Filter=C
BaseClass=CListView
VirtualFilter=VWC
LastObject=CSimulateCommsysView


[CLS:CMainFrame]
Type=0
HeaderFile=MainFrm.h
ImplementationFile=MainFrm.cpp
Filter=T
LastObject=CMainFrame




[CLS:CAboutDlg]
Type=0
HeaderFile=SimulateCommsys.cpp
ImplementationFile=SimulateCommsys.cpp
Filter=D

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=4
Control1=IDC_STATIC,static,1342177283
Control2=IDC_STATIC,static,1342308480
Control3=IDC_STATIC,static,1342308352
Control4=IDOK,button,1342373889

[MNU:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_FILE_SAVE_AS
Command5=ID_FILE_PRINT
Command6=ID_FILE_PRINT_PREVIEW
Command7=ID_FILE_PRINT_SETUP
Command8=ID_FILE_MRU_FILE1
Command9=ID_APP_EXIT
Command10=ID_EDIT_UNDO
Command11=ID_EDIT_CUT
Command12=ID_EDIT_COPY
Command13=ID_EDIT_PASTE
Command14=ID_SYSTEM_CODEC
Command15=ID_SYSTEM_PUNCTURING
Command16=ID_SYSTEM_MODULATION
Command17=ID_SYSTEM_CHANNEL
Command18=ID_SIMULATION_ACCURACY
Command19=ID_SIMULATION_RANGE
Command20=ID_SIMULATION_START
Command21=ID_SIMULATION_STOP
Command22=ID_SIMULATION_BENCHMARK
Command23=ID_VIEW_TOOLBAR
Command24=ID_VIEW_STATUS_BAR
Command25=ID_APP_ABOUT
CommandCount=25

[ACL:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_FILE_PRINT
Command5=ID_EDIT_UNDO
Command6=ID_EDIT_CUT
Command7=ID_EDIT_COPY
Command8=ID_EDIT_PASTE
Command9=ID_EDIT_UNDO
Command10=ID_EDIT_CUT
Command11=ID_EDIT_COPY
Command12=ID_EDIT_PASTE
Command13=ID_NEXT_PANE
Command14=ID_PREV_PANE
CommandCount=14

[TB:IDR_MAINFRAME]
Type=1
Class=?
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_EDIT_CUT
Command5=ID_EDIT_COPY
Command6=ID_EDIT_PASTE
Command7=ID_FILE_PRINT
Command8=ID_APP_ABOUT
CommandCount=8

[DLG:IDD_CHANNEL]
Type=1
Class=CSelectChannelDlg
ControlCount=4
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_TYPE,combobox,1344339971

[CLS:CSelectChannelDlg]
Type=0
HeaderFile=SelectChannelDlg.h
ImplementationFile=SelectChannelDlg.cpp
BaseClass=CDialog
Filter=D
VirtualFilter=dWC
LastObject=CSelectChannelDlg

[DLG:IDD_MODULATOR]
Type=1
Class=CSelectModulatorDlg
ControlCount=4
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_TYPE,combobox,1344339971

[CLS:CSelectModulatorDlg]
Type=0
HeaderFile=SelectModulatorDlg.h
ImplementationFile=SelectModulatorDlg.cpp
BaseClass=CDialog
Filter=D
VirtualFilter=dWC

[DLG:IDD_ACCURACY]
Type=1
Class=CSelectAccuracyDlg
ControlCount=6
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_CONFIDENCE,edit,1350631552
Control4=IDC_ACCURACY,edit,1350631552
Control5=IDC_STATIC,static,1342308352
Control6=IDC_STATIC,static,1342308352

[DLG:IDD_RANGE]
Type=1
Class=CSelectRangeDlg
ControlCount=8
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_SNR_MIN,edit,1350631552
Control4=IDC_SNR_MAX,edit,1350631552
Control5=IDC_SNR_STEP,edit,1350631552
Control6=IDC_STATIC,static,1342308352
Control7=IDC_STATIC,static,1342308352
Control8=IDC_STATIC,static,1342308352

[CLS:CSelectRangeDlg]
Type=0
HeaderFile=SelectRangeDlg.h
ImplementationFile=SelectRangeDlg.cpp
BaseClass=CDialog
Filter=D
LastObject=CSelectRangeDlg
VirtualFilter=dWC

[CLS:CSelectAccuracyDlg]
Type=0
HeaderFile=SelectAccuracyDlg.h
ImplementationFile=SelectAccuracyDlg.cpp
BaseClass=CDialog
Filter=D
LastObject=ID_FILE_SAVE
VirtualFilter=dWC

[DLG:IDD_BENCHMARK]
Type=1
Class=CBenchmarkDlg
ControlCount=27
Control1=IDCANCEL,button,1342242816
Control2=IDC_STATIC,static,1342308352
Control3=IDC_STATIC,static,1342308352
Control4=IDC_STATIC,static,1342308352
Control5=IDC_STATIC,static,1342308352
Control6=IDC_STATIC,static,1342308352
Control7=IDC_TIME,edit,1350631552
Control8=IDC_STATIC,static,1342308352
Control9=IDC_SNR,edit,1350631552
Control10=IDC_PROGRESS,msctls_progress32,1342177281
Control11=IDC_STATIC,static,1342308352
Control12=IDOK,button,1342242816
Control13=IDC_STATIC,static,1342308352
Control14=IDC_STATIC,static,1342308352
Control15=IDC_STATIC,static,1342308352
Control16=IDC_STATIC,static,1342308352
Control17=IDC_PUNCTURE,static,1342312960
Control18=IDC_MODULATOR,static,1342312960
Control19=IDC_CHANNEL,static,1342312960
Control20=IDC_STATIC,button,1342177287
Control21=IDC_STATIC,button,1342177287
Control22=IDC_STATIC,button,1342177287
Control23=IDC_BER,static,1342312960
Control24=IDC_ELAPSED,static,1342312960
Control25=IDC_FRAMES,static,1342312960
Control26=IDC_SPEED,static,1342312960
Control27=IDC_CODEC,edit,1344342084

[CLS:CBenchmarkDlg]
Type=0
HeaderFile=BenchmarkDlg.h
ImplementationFile=BenchmarkDlg.cpp
BaseClass=CDialog
Filter=D
LastObject=CBenchmarkDlg
VirtualFilter=dWC

