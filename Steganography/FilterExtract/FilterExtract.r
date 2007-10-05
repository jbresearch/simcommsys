#include "PIGeneral.h"
#include "PIActions.h"
#include "ScriptingKeys.h"

#define vendorName "Steganography"
#define plugInName "Extract File"
#ifdef NDEBUG
#   define vendorFullName vendorName
#   define plugInFullName plugInName
#else
#   define vendorFullName vendorName " (debug)"
#   define plugInFullName plugInName " (debug)"
#endif

#define plugInSuiteID suiteSteganography
#define plugInEventID eventFilterExtract
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
            /* embedding system */
            "embedding seed",             /* parameter name */
            keyEmbedSeed,                 /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "embedding seed",             /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "embedding rate",             /* parameter name */
            keyEmbedRate,                 /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "embedding rate",             /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "embedding strength",         /* parameter name */
            keyEmbedStrength,             /* parameter key ID */
            typeFloat,                    /* parameter type ID */
            "embedding strength",         /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "preset strength",            /* parameter name */
            keyPresetStrength,            /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "preset strength",            /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* channel interleaver */
            "interleave",                 /* parameter name */
            keyInterleave,                /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "interleave",                 /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "interleaver seed",           /* parameter name */
            keyInterleaverSeed,           /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "interleaver seed",           /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "interleaver density",        /* parameter name */
            keyInterleaverDensity,        /* parameter key ID */
            typeFloat,                    /* parameter type ID */
            "interleaver density",        /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* source data */
            "source type",                /* parameter name */
            keySourceType,                /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "source type",                /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "source seed",                /* parameter name */
            keySourceSeed,                /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "source seed",                /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "source",                     /* parameter name */
            keySource,                    /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "source",                     /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* codec and puncture pattern */
            "codec",                      /* parameter name */
            keyCodec,                     /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "codec file",                 /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "puncture",                   /* parameter name */
            keyPuncture,                  /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "puncturing pattern file",    /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* results storage */
            "results",                    /* parameter name */
            keyResults,                   /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "results file",               /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "embedded image",             /* parameter name */
            keyEmbeddedImage,             /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "embedded image file",        /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "extracted image",            /* parameter name */
            keyExtractedImage,            /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "extracted image file",       /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "embedded",                   /* parameter name */
            keyEmbedded,                  /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "embedded file",              /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "extracted",                  /* parameter name */
            keyExtracted,                 /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "extracted file",             /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "uniform",                    /* parameter name */
            keyUniform,                   /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "uniform file",               /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "decoded",                    /* parameter name */
            keyDecoded,                   /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "decoded file",               /* optional description */
            flagsSingleParameter,         /* parameter flags */

            /* channel parameter computation / feedback */
            "print BER",                  /* parameter name */
            keyPrintBER,                  /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "print BER",                  /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "print SNR",                  /* parameter name */
            keyPrintSNR,                  /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "print SNR",                  /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "print estimate",             /* parameter name */
            keyPrintEstimate,             /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "print estimate",             /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "print chi square",           /* parameter name */
            keyPrintChiSquare,            /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "print chi square",           /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "feedback",                   /* parameter name */
            keyFeedback,                  /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "feedback",                   /* optional description */
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
