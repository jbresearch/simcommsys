#include "PIGeneral.h"
#include "PIActions.h"
#include "ScriptingKeys.h"

#define vendorName "Steganography"
#define plugInName "Wavelet Filter"
#ifdef NDEBUG
#   define vendorFullName vendorName
#   define plugInFullName plugInName
#else
#   define vendorFullName vendorName " (debug)"
#   define plugInFullName plugInName " (debug)"
#endif

#define plugInSuiteID suiteSteganography
#define plugInEventID eventFilterWavelet
#define plugInClassID plugInSuiteID
#define plugInUniqueID ""

resource 'PiPL' ( ResourceID, plugInFullName " PiPL", purgeable )
{
	{
		Kind { Filter },
		Name { plugInFullName "..." },
      Category { vendorFullName },
		Version { (latestFilterVersion << 16) | latestFilterSubVersion },

		CodeWin32X86 { "PluginMain" },

		HasTerminology
		{
			plugInClassID,
			plugInEventID,
			ResourceID,
			plugInUniqueID
		},
		
		SupportedModes
		{
			noBitmap, doesSupportGrayScale,
			noIndexedColor, doesSupportRGBColor,
			doesSupportCMYKColor, doesSupportHSLColor,
			doesSupportHSBColor, doesSupportMultichannel,
			doesSupportDuotone, doesSupportLABColor
		},
			
		EnableInfo
		{
			"in (PSHOP_ImageMode, GrayScaleMode, RGBMode,"
			"CMYKMode, HSLMode, HSBMode, MultichannelMode,"
			"DuotoneMode, LabMode,"
			"Gray16Mode, RGB48Mode, CMYK64Mode, Lab48Mode)"
		},
		
		FilterCaseInfo {
			{	/* array: 7 elements */
				/* Flat data, no selection */
				inStraightData,
				outStraightData,
				doNotWriteOutsideSelection,
				doesNotFilterLayerMasks,
				doesNotWorkWithBlankData,
				doNotCopySourceToDestination,
				/* Flat data with selection */
				inStraightData,
				outStraightData,
				doNotWriteOutsideSelection,
				doesNotFilterLayerMasks,
				doesNotWorkWithBlankData,
				doNotCopySourceToDestination,
				/* Floating selection */
				inStraightData,
				outStraightData,
				doNotWriteOutsideSelection,
				doesNotFilterLayerMasks,
				doesNotWorkWithBlankData,
				doNotCopySourceToDestination,
				/* Editable transparency, no selection */
				inStraightData,
				outStraightData,
				doNotWriteOutsideSelection,
				doesNotFilterLayerMasks,
				doesNotWorkWithBlankData,
				doNotCopySourceToDestination,
				/* Editable transparency, with selection */
				inStraightData,
				outStraightData,
				doNotWriteOutsideSelection,
				doesNotFilterLayerMasks,
				doesNotWorkWithBlankData,
				doNotCopySourceToDestination,
				/* Preserved transparency, no selection */
				inStraightData,
				outStraightData,
				doNotWriteOutsideSelection,
				doesNotFilterLayerMasks,
				doesNotWorkWithBlankData,
				doNotCopySourceToDestination,
				/* Preserved transparency, with selection */
				inStraightData,
				outStraightData,
				doNotWriteOutsideSelection,
				doesNotFilterLayerMasks,
				doesNotWorkWithBlankData,
				doNotCopySourceToDestination
			}
		}
	}
};

resource 'aete' ( ResourceID, plugInFullName " Dictionary", purgeable )
{
   1, 0, english, roman,                  /* aete version and language specifiers */
   {
      vendorFullName,                     /* vendor suite name */
      vendorFullName " Software",         /* optional description */
      plugInSuiteID,                      /* suite ID */
      1,                                  /* suite code, must be 1 */
      1,                                  /* suite level, must be 1 */
      {  /* structure for filters */
         plugInFullName,                  /* unique filter name */
         plugInFullName,                  /* optional description */
         plugInClassID,                   /* class ID, must be unique or Suite ID */
         plugInEventID,                   /* event ID, must be unique to class ID */
         
         NO_REPLY,                        /* never a reply */
         IMAGE_DIRECT_PARAMETER,          /* direct parameter - see PIActions.h for more */
         {  /* parameters here, if any */
            /* wavelet basis */
            "wavelet type",               /* parameter name */
            keyWaveletType,               /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "wavelet type",               /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "wavelet parameter",          /* parameter name */
            keyWaveletPar,                /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "wavelet parameter",          /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "wavelet level",              /* parameter name */
            keyWaveletLevel,              /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "wavelet level",              /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* thresholding */
            "threshold type",             /* parameter name */
            keyThreshType,                /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "threshold type",             /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "threshold selector",         /* parameter name */
            keyThreshSelector,            /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "threshold selector",         /* optional description */
            flagsSingleParameter,         /* parameter flags */
            
            "threshold cutoff",           /* parameter name */
            keyThreshCutoff,              /* parameter key ID */
            typeFloat,                    /* parameter type ID */
            "threshold cutoff",           /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* tiling */
            "tile width",                 /* parameter name */
            keyTileX,                     /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "tile width",                 /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "tile height",                /* parameter name */
            keyTileY,                     /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "tile height",                /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "whole image",                /* parameter name */
            keyWholeImage,                /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "operate on whole image",     /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* other */
            "keep noise",                 /* parameter name */
            keyKeepNoise,                 /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "keep noise",                 /* optional description */
            flagsSingleParameter          /* parameter flags */
         }
      },
      {  /* non-filter plug-in class here */
      },
      {  /* comparison ops (not supported) */
      },
      {  /* any enumerations */
      }
   }
};
