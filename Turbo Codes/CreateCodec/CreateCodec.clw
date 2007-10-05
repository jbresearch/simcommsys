; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CSelectInterleaverDlg
LastTemplate=CDialog
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "CreateCodec.h"
LastPage=0

ClassCount=11
Class1=CCreateCodecApp
Class2=CCreateCodecDoc
Class3=CCreateCodecView
Class4=CMainFrame

ResourceCount=8
Resource1=IDD_ABOUTBOX
Class5=CAboutDlg
Resource2=IDD_INT
Class6=CSelectCodecDlg
Resource3=IDD_BOOL
Resource4=IDD_INTERLEAVER
Class7=CSelectIntDlg
Class8=CSelectBoolDlg
Resource5=IDD_GENERATOR
Class9=CSelectEncoderDlg
Resource6=IDD_CODEC
Class10=CSelectGeneratorDlg
Resource7=IDD_ENCODER
Class11=CSelectInterleaverDlg
Resource8=IDR_MAINFRAME

[CLS:CCreateCodecApp]
Type=0
HeaderFile=CreateCodec.h
ImplementationFile=CreateCodec.cpp
Filter=N

[CLS:CCreateCodecDoc]
Type=0
HeaderFile=CreateCodecDoc.h
ImplementationFile=CreateCodecDoc.cpp
Filter=N
LastObject=65535

[CLS:CCreateCodecView]
Type=0
HeaderFile=CreateCodecView.h
ImplementationFile=CreateCodecView.cpp
Filter=C
BaseClass=CTreeView
VirtualFilter=VWC
LastObject=CCreateCodecView


[CLS:CMainFrame]
Type=0
HeaderFile=MainFrm.h
ImplementationFile=MainFrm.cpp
Filter=T




[CLS:CAboutDlg]
Type=0
HeaderFile=CreateCodec.cpp
ImplementationFile=CreateCodec.cpp
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
Command14=ID_VIEW_TOOLBAR
Command15=ID_VIEW_STATUS_BAR
Command16=ID_APP_ABOUT
CommandCount=16

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

[DLG:IDD_CODEC]
Type=1
Class=CSelectCodecDlg
ControlCount=6
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_TYPE,combobox,1344339971
Control5=IDC_STATIC,static,1342308352
Control6=IDC_MATH,combobox,1344339971

[CLS:CSelectCodecDlg]
Type=0
HeaderFile=SelectCodecDlg.h
ImplementationFile=SelectCodecDlg.cpp
BaseClass=CDialog
Filter=D
LastObject=CSelectCodecDlg
VirtualFilter=dWC

[DLG:IDD_INT]
Type=1
Class=CSelectIntDlg
ControlCount=4
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_VALUE,edit,1350639744
Control4=IDC_STATIC,static,1342308352

[DLG:IDD_BOOL]
Type=1
Class=CSelectBoolDlg
ControlCount=5
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_NAY,button,1342308361
Control5=IDC_AYE,button,1342177289

[CLS:CSelectIntDlg]
Type=0
HeaderFile=SelectIntDlg.h
ImplementationFile=SelectIntDlg.cpp
BaseClass=CDialog
Filter=D
LastObject=CSelectIntDlg
VirtualFilter=dWC

[CLS:CSelectBoolDlg]
Type=0
HeaderFile=SelectBoolDlg.h
ImplementationFile=SelectBoolDlg.cpp
BaseClass=CDialog
Filter=D
LastObject=CSelectBoolDlg
VirtualFilter=dWC

[DLG:IDD_ENCODER]
Type=1
Class=CSelectEncoderDlg
ControlCount=4
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_TYPE,combobox,1344339971

[CLS:CSelectEncoderDlg]
Type=0
HeaderFile=SelectEncoderDlg.h
ImplementationFile=SelectEncoderDlg.cpp
BaseClass=CDialog
Filter=D
VirtualFilter=dWC

[DLG:IDD_GENERATOR]
Type=1
Class=CSelectGeneratorDlg
ControlCount=8
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_VALUE,edit,1350639744
Control4=IDC_STATIC,static,1342308352
Control5=IDC_STATIC,static,1342308352
Control6=IDC_STATIC,static,1342308352
Control7=IDC_INPUT,msctls_trackbar32,1342242821
Control8=IDC_OUTPUT,msctls_trackbar32,1342242821

[CLS:CSelectGeneratorDlg]
Type=0
HeaderFile=SelectGeneratorDlg.h
ImplementationFile=SelectGeneratorDlg.cpp
BaseClass=CDialog
Filter=D
VirtualFilter=dWC
LastObject=CSelectGeneratorDlg

[DLG:IDD_INTERLEAVER]
Type=1
Class=CSelectInterleaverDlg
ControlCount=4
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_TYPE,combobox,1344339971

[CLS:CSelectInterleaverDlg]
Type=0
HeaderFile=SelectInterleaverDlg.h
ImplementationFile=SelectInterleaverDlg.cpp
BaseClass=CDialog
Filter=D
VirtualFilter=dWC
LastObject=IDC_TYPE

