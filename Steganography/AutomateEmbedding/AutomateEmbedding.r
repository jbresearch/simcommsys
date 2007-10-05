#include "PIGeneral.h"
#include "PIActions.h"
#include "ScriptingKeys.h"

#define vendorName "Steganography"
#define plugInName "Embedding"
#ifdef NDEBUG
#   define vendorFullName vendorName
#   define plugInFullName plugInName
#else
#   define vendorFullName vendorName " (debug)"
#   define plugInFullName plugInName " (debug)"
#endif

#define plugInSuiteID suiteSteganography
#define plugInEventID eventAutomateEmbedding
#define plugInClassID plugInSuiteID
#define plugInUniqueID ""

resource 'PiPL' ( ResourceID, plugInFullName " PiPL", purgeable )
{
	{
		Kind { Actions },
		Name { plugInFullName "..." },
      Category { vendorFullName },
		Version { (latestActionsPlugInVersion << 16) | latestActionsPlugInSubVersion },

		CodeWin32X86 { "PluginMain" },

		HasTerminology
		{
			plugInClassID,
			plugInEventID,
			ResourceID,
			plugInUniqueID
		},
		
      // if specified, enables plug-in when there's open document of that type,
      // otherwise plug-in always enabled
		EnableInfo
		{
			"in (PSHOP_ImageMode, GrayScaleMode, RGBMode,"
			"CMYKMode, HSLMode, HSBMode, MultichannelMode,"
			"DuotoneMode, LabMode,"
			"Gray16Mode, RGB48Mode, CMYK64Mode, Lab48Mode)"
		},
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
            // path for output files
            "output",                     /* parameter name */
            keyOutput,                    /* parameter key ID */
            typeChar,                     /* parameter type ID */
            "output path",                /* optional description */
            flagsSingleParameter,         /* parameter flags */

            // system options
            "jpeg",                       /* parameter name */
            keyJpeg,                      /* parameter key ID */
            typeBoolean,                  /* parameter type ID */
            "simulate jpeg compression",  /* optional description */
            flagsSingleParameter,         /* parameter flags */

            // range of embedding strengths
            "embedding strength min",     /* parameter name */
            keyStrengthMin,               /* parameter key ID */
            typeFloat,                    /* parameter type ID */
            "embedding strength minimum", /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "embedding strength max",     /* parameter name */
            keyStrengthMax,               /* parameter key ID */
            typeFloat,                    /* parameter type ID */
            "embedding strength maximum", /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "embedding strength step",    /* parameter name */
            keyStrengthStep,              /* parameter key ID */
            typeFloat,                    /* parameter type ID */
            "embedding strength step",    /* optional description */
            flagsSingleParameter,         /* parameter flags */

            // range of JPEG compression quality (if requested)
            "jpeg quality min",           /* parameter name */
            keyJpegMin,                   /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "jpeg quality minimum",       /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "jpeg quality max",           /* parameter name */
            keyJpegMax,                   /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "jpeg quality maximum",       /* optional description */
            flagsSingleParameter,         /* parameter flags */

            "jpeg quality step",          /* parameter name */
            keyJpegStep,                  /* parameter key ID */
            typeInteger,                  /* parameter type ID */
            "jpeg quality step size",     /* optional description */
            flagsSingleParameter,         /* parameter flags */
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
