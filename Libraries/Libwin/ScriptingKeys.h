#ifndef __scriptingkeys_h
#define __scriptingkeys_h

// *** Global definitions
#define ResourceID 16000


// *** Suite IDs
#ifdef NDEBUG
#   define suiteSteganography    'steN'
#   define suiteBinary           'binN'
#else
#   define suiteSteganography    'steD'
#   define suiteBinary           'binD'
#endif


// *** Event IDs
#ifdef NDEBUG
#   define eventFilterATM        'atmN'
#   define eventFilterAW         'wieN'
#   define eventFilterEmbed      'embN'
#   define eventFilterEnergy     'eneN'
#   define eventFilterExport     'expN'
#   define eventFilterExtract    'extN'
#   define eventFilterLevels     'levN'
#   define eventFilterOrphans    'orpN'
#   define eventFilterVariance   'varN'
#   define eventFilterWavelet    'wavN'
#   define eventAutomateGraphing 'graN'
#   define eventAutomateEmbedding 'emaN'
#else
#   define eventFilterATM        'atmD'
#   define eventFilterAW         'wieD'
#   define eventFilterEmbed      'embD'
#   define eventFilterEnergy     'eneD'
#   define eventFilterExport     'expD'
#   define eventFilterExtract    'extD'
#   define eventFilterLevels     'levD'
#   define eventFilterOrphans    'orpD'
#   define eventFilterVariance   'varD'
#   define eventFilterWavelet    'wavD'
#   define eventAutomateGraphing 'graD'
#   define eventAutomateEmbedding 'emaD'
#endif


// *** Key IDs

// numerical types
#define keyWeight                'weiG'
#define keyAlpha                 'alpH'
#define keyWaveletType           'wavT'
#define keyWaveletPar            'wavP'
#define keyWaveletLevel          'wavL'
#define keyThreshType            'thrT'
#define keyThreshSelector        'thrS'
#define keyThreshCutoff          'thrC'
#define keyTileX                 'tilX'
#define keyTileY                 'tilY'
#define keyEmbedSeed             'embS'
#define keyEmbedRate             'embR'
#define keyEmbedStrength         'embT'
#define keyInterleaverSeed       'intS'
#define keyInterleaverDensity    'intD'
#define keySourceType            'souT'
#define keySourceSeed            'souS'
#define keyFeedback              'feeD'
#define keyStrengthMin           'smiN'
#define keyStrengthMax           'smaX'
#define keyStrengthStep          'sstE'
#define keyJpegMin               'jmiN'
#define keyJpegMax               'jmaX'
#define keyJpegStep              'jstE'

// boolean types
#define keyWholeImage            'whoL'
#define keyKeepNoise             'keeP'
#define keyDisplayEnergy         'eneR'
#define keyDisplayVariance       'varI'
#define keyDisplayPixelCount     'pixE'
#define keyInterleave            'intE'
#define keyPresetStrength        'preS'
#define keyPrintBER              'pBER'
#define keyPrintSNR              'pSNR'
#define keyPrintEstimate         'pEST'
#define keyPrintChiSquare        'pCHI'
#define keyJpeg                  'jpeG'

// string types
#define keyFileName              'filE'
#define keyCodec                 'codE'
#define keyResults               'resU'
#define keyEmbeddedImage         'embI'
#define keyExtractedImage        'extI'
#define keyEmbedded              'embE'
#define keyExtracted             'extR'
#define keyUniform               'uniF'
#define keyDecoded               'decO'
#define keyParameters            'parA'

#endif
