import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ActivityIndicator, ScrollView, TouchableOpacity, Platform, Alert } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ort from 'onnxruntime-react-native';
import dataset from '../assets/dataset/parquet_embeds_rag.json'

// Define model options with optimization variants
const MODEL_OPTIONS = [
  { 
    label: "Custom ONNX (eldoon101/idk-parahrase-miniLM-onnxver)",
    repo: "eldoon101/idk-parahrase-miniLM-onnxver",
    fileName: "paraphrase_multilingual_miniLM_L12_v2.onnx",
    tokenPadding: 128,
    description: "Custom implementation",
    needsTokenTypeIds: false
  },
  {
    label: "Official onnx/model.onnx",
    repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fileName: "onnx/model.onnx",
    tokenPadding: 12,
    description: "Standard ONNX version",
    needsTokenTypeIds: true
  },
  {
    label: "Optimized Level 1",
    repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fileName: "onnx/model_O1.onnx",
    tokenPadding: 12,
    description: "Optimization level 1",
    needsTokenTypeIds: true
  },
  {
    label: "Optimized Level 2",
    repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fileName: "onnx/model_O2.onnx",
    tokenPadding: 12,
    description: "Optimization level 2",
    needsTokenTypeIds: true
  },
  {
    label: "Optimized Level 3",
    repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fileName: "onnx/model_O3.onnx",
    tokenPadding: 12,
    description: "Optimization level 3",
    needsTokenTypeIds: true
  },
  {
    label: "Optimized Level 4",
    repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fileName: "onnx/model_O4.onnx",
    tokenPadding: 12,
    description: "Optimization level 4 (half precision)",
    needsTokenTypeIds: true
  }
];

// Add platform-specific quantized models
if (Platform.OS === 'ios') {
  MODEL_OPTIONS.push({
    label: "Quantized for ARM64 (iOS)",
    repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fileName: "onnx/model_qint8_arm64.onnx",
    tokenPadding: 12,
    description: "8-bit quantized for ARM64 processors",
    needsTokenTypeIds: true
  });
} else {
  // For Android and other platforms, add the AVX2 version as it's more broadly compatible
  MODEL_OPTIONS.push({
    label: "Quantized (General)",
    repo: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    fileName: "onnx/model_quint8_avx2.onnx",
    tokenPadding: 12,
    description: "8-bit quantized for general use",
    needsTokenTypeIds: true
  });
}

// Replace the existing SAMPLE_TEXT constant and add a pre-computed embedding sample
const SAMPLE_TEXT = "human: Apakah latar belakang Dato' Sri Mohd Najib bin Tun Abdul Razak? gpt: Dato' Sri Mohd Najib bin Tun Abdul Razak dilahirkan pada 23 Julai 1953 di Kuala Lipis, Pahang. Beliau telah berkahwin dengan Datin Sri Rosmah binti Mansor dan terkenal kerana memperkenalkan Gagasan 1 Malaysia. Sebagai pengundi yang berpengaruh dalam politik Malaysia, Najib mempunyai latar belakang yang kuat dalam bidang pendidikan, mendapatkan pendidikan rendah dan menengah di St. John Institution, Kuala Lumpur, serta melanjutkan pengajian di Universiti Nottingham, England, dalam bidang ekonomi industri.";

// Pre-computed embedding for the sample text (Malaysian political figure)
const SAMPLE_EMBEDDING = [
  0.043607648462057114,
  0.47821709513664246,
  -0.09822244197130203,
  0.037893835455179214,
  0.23725835978984833,
  0.19291339814662933,
  0.1551094949245453,
  0.06336215883493423,
  0.1408117115497589,
  0.19386734068393707,
  0.18164785206317902,
  -0.37161538004875183,
  -0.030460262671113014,
  0.08479563146829605,
  0.06540091335773468,
  0.01084422878921032,
  -0.2947326600551605,
  -0.2140970528125763,
  0.2597017288208008,
  -0.11500181257724762,
  -0.1737183779478073,
  -0.06610049307346344,
  0.058403756469488144,
  -0.08035694062709808,
  0.05126136168837547,
  -0.028716307133436203,
  0.4800509810447693,
  0.049628932029008865,
  0.016745150089263916,
  0.08064240217208862,
  0.1310197114944458,
  -0.03585123270750046,
  0.03920236602425575,
  0.04088423773646355,
  -0.1422276645898819,
  0.1453528106212616,
  0.04810035601258278,
  0.3183874487876892,
  0.07819227129220963,
  -0.0740414410829544,
  0.07818568497896194,
  -0.3966996669769287,
  0.2810158431529999,
  -0.07167242467403412,
  0.17730562388896942,
  0.1543150693178177,
  -0.04690581560134888,
  0.12722812592983246,
  -0.29110971093177795,
  0.1722564399242401,
  -0.009259081445634365,
  -0.15308713912963867,
  -0.11470774561166763,
  0.03955954313278198,
  0.1121104508638382,
  0.17488333582878113,
  0.08009106665849686,
  0.17677156627178192,
  0.055915672332048416,
  -0.12209366261959076,
  -0.11362908780574799,
  0.3229503333568573,
  -0.30348116159439087,
  -0.0393265075981617,
  -0.24909064173698425,
  -0.18991868197917938,
  0.23306740820407867,
  -0.17739582061767578,
  -0.08570309728384018,
  0.08074825257062912,
  0.15300044417381287,
  -0.02603657729923725,
  0.20956242084503174,
  -0.008700565434992313,
  -0.17614510655403137,
  0.108340322971344,
  -0.03102358803153038,
  0.1388801783323288,
  -0.10627644509077072,
  0.09583967179059982,
  -0.004806957207620144,
  0.18522129952907562,
  -0.09504956752061844,
  -0.22521798312664032,
  -0.18127906322479248,
  0.0783068910241127,
  -0.29110100865364075,
  -0.205466166138649,
  -0.0485113300383091,
  -0.02226039581000805,
  0.0499994121491909,
  -0.1476271152496338,
  0.003682076930999756,
  -0.20050311088562012,
  0.16483400762081146,
  0.03693883493542671,
  0.1432041972875595,
  0.204811692237854,
  -0.06688155978918076,
  0.2199123054742813,
  -0.007777565158903599,
  0.19186054170131683,
  0.006247829645872116,
  0.21512556076049805,
  -0.14428158104419708,
  0.16748347878456116,
  0.00542467599734664,
  0.14866553246974945,
  0.04607611149549484,
  0.06582609564065933,
  -0.225162073969841,
  0.08278922736644745,
  -0.20782698690891266,
  -0.19437281787395477,
  0.0017887966241687536,
  -0.18923726677894592,
  -0.4445427358150482,
  0.07217422872781754,
  -0.2580574154853821,
  -0.08482667058706284,
  0.0944342091679573,
  0.032088685780763626,
  -0.20422227680683136,
  -0.4298813045024872,
  -0.01017827820032835,
  0.11098595708608627,
  0.01249412540346384,
  0.17620393633842468,
  -0.01719796657562256,
  -0.20803925395011902,
  0.08275182545185089,
  0.08418089151382446,
  0.09807096421718597,
  0.22843630611896515,
  0.15564849972724915,
  -0.25020548701286316,
  0.114876389503479,
  0.14916332066059113,
  -0.09157969802618027,
  -0.2492767870426178,
  -0.08635091781616211,
  -0.2790702283382416,
  -0.10054036229848862,
  -0.0019307972397655249,
  0.05645919218659401,
  -0.09322142601013184,
  0.018190991133451462,
  0.3001042902469635,
  -0.014109672047197819,
  -0.19831643998622894,
  0.09122510999441147,
  -0.2433079034090042,
  -0.30157938599586487,
  0.08632636815309525,
  0.19112154841423035,
  0.008844112046062946,
  0.14195039868354797,
  0.19826267659664154,
  -0.06793917715549469,
  -0.07308372110128403,
  -0.14593307673931122,
  -0.16047099232673645,
  -0.15718436241149902,
  -0.04648677632212639,
  -0.05253412947058678,
  -0.2972610890865326,
  0.17859390377998352,
  -0.2030099779367447,
  0.11327509582042694,
  -0.02306324616074562,
  0.013891850598156452,
  0.20022113621234894,
  -0.17748917639255524,
  0.04313722997903824,
  0.18840396404266357,
  -0.16996847093105316,
  -0.038239333778619766,
  0.11109703034162521,
  -0.04105282202363014,
  -0.2286117821931839,
  -0.20396780967712402,
  -0.026148106902837753,
  -0.04903571680188179,
  0.021654149517416954,
  0.1593993902206421,
  -0.08234795182943344,
  -0.01845654845237732,
  0.2760382294654846,
  -0.08283495903015137,
  0.1582016795873642,
  -0.056026410311460495,
  0.08348852396011353,
  -0.0110709760338068,
  -0.4096004068851471,
  0.11697708070278168,
  0.030316269025206566,
  0.22691985964775085,
  -0.09711132943630219,
  0.31157901883125305,
  0.01570320688188076,
  -0.23396511375904083,
  0.006685387343168259,
  -0.3071431517601013,
  -0.034273725003004074,
  0.20834049582481384,
  -0.010793253779411316,
  0.1536957174539566,
  0.19775749742984772,
  -0.006290886551141739,
  0.3583829402923584,
  0.003575905691832304,
  0.022903144359588623,
  -0.050353698432445526,
  -0.04772549122571945,
  0.10482200235128403,
  -0.10814748704433441,
  0.13955940306186676,
  0.02060401253402233,
  0.15902085602283478,
  0.03476005420088768,
  0.1599784940481186,
  -0.32810884714126587,
  -0.23874743282794952,
  -0.17990221083164215,
  -0.0795087143778801,
  0.03776485472917557,
  -0.022406816482543945,
  -0.18416862189769745,
  0.1499396115541458,
  0.0047006974928081036,
  0.19994597136974335,
  0.19696788489818573,
  -0.06821215897798538,
  -0.3377523422241211,
  -0.09139399230480194,
  -0.06109872832894325,
  0.3082441985607147,
  -0.11376232653856277,
  0.046673260629177094,
  0.019334252923727036,
  0.135398268699646,
  0.026801377534866333,
  0.09879127889871597,
  0.20942378044128418,
  0.23954296112060547,
  -0.12831197679042816,
  -0.19713525474071503,
  0.26799261569976807,
  0.0864674374461174,
  0.11043904721736908,
  -0.09601198136806488,
  -0.08348363637924194,
  -0.06514457613229752,
  0.3438687324523926,
  0.00896210316568613,
  -0.038702283054590225,
  -0.09985653311014175,
  -0.04769599810242653,
  0.04529228433966637,
  -0.2999074161052704,
  -0.006008889526128769,
  0.013170747086405754,
  0.053742777556180954,
  -0.055427201092243195,
  -0.000431513151852414,
  0.027957627549767494,
  0.004590548574924469,
  0.09031976759433746,
  0.01802162639796734,
  0.13751277327537537,
  -0.23674249649047852,
  0.026442578062415123,
  0.18977090716362,
  -0.2144109308719635,
  -0.11909493058919907,
  -0.03754763677716255,
  0.1486850082874298,
  -0.23226897418498993,
  0.17660173773765564,
  0.08123449236154556,
  0.11969534307718277,
  -0.08145279437303543,
  0.5082626938819885,
  -0.17411112785339355,
  -0.10788873583078384,
  0.0022423635236918926,
  0.0666174441576004,
  0.05263848975300789,
  -0.17781448364257812,
  0.25517910718917847,
  0.08309756219387054,
  0.06933125108480453,
  0.03437458723783493,
  -0.3017306327819824,
  0.15407955646514893,
  -0.13333909213542938,
  -0.002300441265106201,
  0.026785364374518394,
  -0.1689787656068802,
  0.047161754220724106,
  -0.0902455672621727,
  0.0474742166697979,
  0.24490539729595184,
  -0.022369537502527237,
  0.1748131364583969,
  0.2578946650028229,
  -0.13198040425777435,
  -0.21467842161655426,
  -0.11832237243652344,
  0.03852568939328194,
  0.03858068957924843,
  -0.3379654884338379,
  0.054669398814439774,
  -0.37994351983070374,
  -0.09437025338411331,
  0.1792570799589157,
  0.0007468951516784728,
  -0.002942219376564026,
  0.24491426348686218,
  -0.2107284963130951,
  -0.3015386164188385,
  -0.4120219349861145,
  0.0031528938561677933,
  -0.046516768634319305,
  -0.038322813808918,
  0.1876859962940216,
  -0.10636555403470993,
  -0.2022004872560501,
  0.2844804525375366,
  0.16589298844337463,
  -0.02961036004126072,
  0.04448872059583664,
  0.0689149871468544,
  0.013856664299964905,
  0.18147757649421692,
  -0.08849040418863297,
  0.20489844679832458,
  0.0026245771441608667,
  0.005713160615414381,
  -0.18892867863178253,
  0.09106606990098953,
  0.08097703009843826,
  0.11500483006238937,
  0.12811025977134705,
  0.17708618938922882,
  0.0176268108189106,
  -0.09612768888473511,
  -0.052534181624650955,
  -0.021578418090939522,
  -0.1384323686361313,
  -0.08432397991418839,
  0.11598189175128937,
  -0.04066271334886551,
  -0.15495602786540985,
  0.09235557168722153,
  -0.15135271847248077,
  -0.13865673542022705,
  0.08514729887247086,
  0.003695302875712514,
  -0.061120565980672836,
  -0.12953785061836243,
  -0.09961096942424774,
  0.15079781413078308,
  0.0612313486635685,
  -0.22637417912483215,
  0.11418234556913376,
  -0.10577718168497086,
  0.3667445778846741,
  -0.4075436592102051,
  -0.16323794424533844,
  -0.1728266030550003,
  -0.050084471702575684,
  0.28227463364601135,
  -0.1027442067861557,
  -0.042384326457977295,
  0.09207213670015335,
  0.12717105448246002,
  -0.14688418805599213,
  0.04413944110274315,
  0.11274327337741852,
  0.23782242834568024,
  0.1788223534822464,
  -0.22290228307247162,
  -0.09208358824253082
];

// Add this debug check to help spot dimension mismatches
console.log(`SAMPLE_EMBEDDING dimensions: ${SAMPLE_EMBEDDING.length}`);

// Add this new function to compare with a pre-computed embedding
const compareWithPrecomputedSample = (userEmbedding: number[] | null) => {
  if (!userEmbedding || userEmbedding.length !== SAMPLE_EMBEDDING.length) {
    return {
      similarity: 0,
      message: "Could not compare embeddings (dimension mismatch)"
    };
  }
  
  const similarityScore = calculateCosineSimilarity(userEmbedding, SAMPLE_EMBEDDING);
  
  let message = "";
  if (similarityScore > 0.8) {
    message = "Very high similarity to the Malaysian political context query";
  } else if (similarityScore > 0.6) {
    message = "Strong similarity to the Malaysian political context query";
  } else if (similarityScore > 0.4) {
    message = "Moderate similarity to the Malaysian political context query";
  } else if (similarityScore > 0.2) {
    message = "Slight similarity to the Malaysian political context query";
  } else {
    message = "Low similarity to the Malaysian political context query";
  }
  
  return { similarity: similarityScore, message };
};

// Utility function to calculate cosine similarity
const calculateCosineSimilarity = (embedding1, embedding2) => {
  if (!embedding1 || !embedding2 || embedding1.length !== embedding2.length) {
    return 0;
  }
  
  // Check for identical arrays first - should be exactly 1.0
  let identical = true;
  for (let i = 0; i < embedding1.length; i++) {
    if (embedding1[i] !== embedding2[i]) {
      identical = false;
      break;
    }
  }
  
  if (identical) return 1.0;
  
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    norm1 += embedding1[i] * embedding1[i];
    norm2 += embedding2[i] * embedding2[i];
  }
  
  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);
  
  if (norm1 === 0 || norm2 === 0) {
    return 0;
  }
  
  // Handle potential floating-point errors - constrain to [-1, 1]
  const similarity = dotProduct / (norm1 * norm2);
  return Math.max(-1, Math.min(1, similarity));
};

// Model Selection Screen component
const ModelSelectionScreen = ({ onModelReady, onBack }) => {
  const [selectedModelIndex, setSelectedModelIndex] = useState(0);
  const [modelLoading, setModelLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<{ size?: string, loadTime?: number, readyToUse?: boolean }>({});
  const [showOptimizedModels, setShowOptimizedModels] = useState(false);
  const [showQuantizedModels, setShowQuantizedModels] = useState(false);
  const [modelSelected, setModelSelected] = useState(false);

  // Effect to update model status when selection changes
  useEffect(() => {
    setModelSelected(true);
    checkModelCache();
  }, [selectedModelIndex]);
  
  // Check if model is already cached
  const checkModelCache = async () => {
    try {
      const selectedModel = MODEL_OPTIONS[selectedModelIndex];
      const modelRepo = selectedModel.repo;
      const modelFileName = selectedModel.fileName;
      
      // Create a sanitized directory name from model path
      const safeModelName = modelFileName.replace(/\//g, '_');
      const modelDir = `${FileSystem.cacheDirectory}models/${modelRepo.replace('/', '_')}/`;
      const modelPath = `${modelDir}${safeModelName}`;
      
      const modelFileInfo = await FileSystem.getInfoAsync(modelPath);
      
      if (modelFileInfo.exists) {
        const modelSize = modelFileInfo.size || 0;
        setModelInfo({
          size: formatFileSize(modelSize),
          readyToUse: false,
          cached: true
        });
      } else {
        setModelInfo({
          readyToUse: false,
          cached: false
        });
      }
    } catch (err) {
      console.error('Error checking model cache:', err);
    }
  };
  
  const loadModel = async () => {
    try {
      const loadStartTime = Date.now();
      setModelLoading(true);
      setError(null);
      
      const selectedModel = MODEL_OPTIONS[selectedModelIndex];
      console.log(`Loading model: ${selectedModel.label}`);

      // Download tokenizer files directly from Hugging Face
      console.log('Downloading tokenizer files...');
      const tokenizerRepo = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';
      const tokenizerFiles = [
        'tokenizer_config.json',
        'tokenizer.json',
        'special_tokens_map.json',
      ];
      
      const tokenizerDir = `${FileSystem.cacheDirectory}tokenizer/`;
      
      // Create directory for tokenizer files
      await FileSystem.makeDirectoryAsync(tokenizerDir, { intermediates: true }).catch(e => {
        console.log('Tokenizer directory creation:', e);
      });
      
      // Download all tokenizer files
      const downloadPromises = tokenizerFiles.map(async (fileName) => {
        const fileUrl = `https://huggingface.co/${tokenizerRepo}/resolve/main/${fileName}`;
        const filePath = `${tokenizerDir}${fileName}`;
        
        // Check if file already exists (cached)
        const fileInfo = await FileSystem.getInfoAsync(filePath);
        
        if (!fileInfo.exists) {
          console.log(`Downloading ${fileName}...`);
          const result = await FileSystem.downloadAsync(fileUrl, filePath);
          
          if (result.status !== 200) {
            throw new Error(`Failed to download ${fileName}: HTTP status ${result.status}`);
          }
        } else {
          console.log(`Using cached ${fileName}`);
        }
        
        return { fileName, path: filePath };
      });
      
      const downloadedFiles = await Promise.all(downloadPromises);
      console.log('Tokenizer files ready:', downloadedFiles.map(f => f.fileName).join(', '));
      
      // Load tokenizer configuration
      const tokenizerConfigPath = `${tokenizerDir}tokenizer_config.json`;
      const tokenizerJsonPath = `${tokenizerDir}tokenizer.json`;
      
      const tokenizerConfigText = await FileSystem.readAsStringAsync(tokenizerConfigPath);
      const tokenizerJsonText = await FileSystem.readAsStringAsync(tokenizerJsonPath);
      const tokenizerConfig = JSON.parse(tokenizerConfigText);
      const tokenizerData = JSON.parse(tokenizerJsonText);
      
      // Extract vocabulary from tokenizer.json
      const vocabulary = Object.keys(tokenizerData.model?.vocab || {});
      console.log(`Extracted vocabulary with ${vocabulary.length} tokens`);
      
      // Create a custom tokenizer implementation
      const customTokenizer = createTokenizer(tokenizerConfig, tokenizerData, selectedModel.tokenPadding, selectedModel.needsTokenTypeIds);
      
      // Download the ONNX model from Hugging Face
      const modelRepo = selectedModel.repo;
      const modelFileName = selectedModel.fileName;
      const modelUrl = `https://huggingface.co/${modelRepo}/resolve/main/${modelFileName}`;

      console.log('Downloading ONNX model from:', modelUrl);
      
      // Create a sanitized directory name from model path
      const safeModelName = modelFileName.replace(/\//g, '_');
      const modelDir = `${FileSystem.cacheDirectory}models/${modelRepo.replace('/', '_')}/`;
      const modelPath = `${modelDir}${safeModelName}`;
      
      // Create directory if it doesn't exist
      await FileSystem.makeDirectoryAsync(modelDir, { intermediates: true }).catch(e => {
        // Directory might already exist, which is fine
        console.log('Directory creation:', e);
      });

      // Check if model is already cached
      const modelFileInfo = await FileSystem.getInfoAsync(modelPath);
      let modelSize = 0;
      
      if (!modelFileInfo.exists) {
        // Download the model file
        console.log('Model not cached, downloading...');
        const downloadResult = await FileSystem.downloadAsync(
          modelUrl,
          modelPath
        );
        
        if (downloadResult.status !== 200) {
          throw new Error(`Failed to download model: HTTP status ${downloadResult.status}`);
        }
        
        // Get file info to show size
        const downloadedFileInfo = await FileSystem.getInfoAsync(modelPath);
        modelSize = downloadedFileInfo.size || 0;
        
        console.log(`Model downloaded successfully (${formatFileSize(modelSize)})`);
      } else {
        modelSize = modelFileInfo.size || 0;
        console.log(`Using cached model (${formatFileSize(modelSize)})`);
      }
      
      // Load ONNX model
      console.log('Loading model into memory...');
      const modelLoadStart = Date.now();
      const session = await ort.InferenceSession.create(modelPath);
      const modelLoadTime = Date.now() - modelLoadStart;

      console.log(`Model loaded successfully in ${modelLoadTime}ms`);
      
      // Update model info for display
      setModelInfo({
        size: formatFileSize(modelSize),
        loadTime: modelLoadTime,
        readyToUse: true,
        cached: modelFileInfo.exists
      });
      
      const totalLoadTime = Date.now() - loadStartTime;
      console.log(`Total resource loading time: ${totalLoadTime}ms`);
      
      setModelLoading(false);
      
      // Pass model data to parent
      onModelReady({
        session,
        tokenizer: customTokenizer,
        modelInfo: {
          name: selectedModel.label,
          description: selectedModel.description,
          size: formatFileSize(modelSize),
          loadTime: modelLoadTime,
          paddingLength: selectedModel.tokenPadding
        }
      });
      
    } catch (err) {
      console.error('Resource loading error:', err);
      if (err instanceof Error) {
        setError('Failed to load resources: ' + err.message);
      } else {
        setError('Failed to load resources: An unknown error occurred');
      }
      setModelLoading(false);
    }
  };

  const confirmModelDownload = () => {
    const selectedModel = MODEL_OPTIONS[selectedModelIndex];
    
    // Check if we should show download size warning for non-cached models
    if (!modelInfo.cached) {
      // For large models (optimized level 1-3 are about 470MB each)
      const isLargeModel = selectedModel.fileName.includes('model_O1') || 
                           selectedModel.fileName.includes('model_O2') || 
                           selectedModel.fileName.includes('model_O3');
                           
      const sizeWarning = isLargeModel 
        ? 'This model is approximately 470MB and may take a few minutes to download.' 
        : '';
      
      Alert.alert(
        'Download Model',
        `Do you want to download and load ${selectedModel.label}?\n${sizeWarning}`,
        [
          {
            text: 'Cancel',
            style: 'cancel'
          },
          {
            text: 'Download & Load',
            onPress: loadModel
          }
        ]
      );
    } else {
      // Model already cached, just load it
      loadModel();
    }
  };

  // Render model selection item
  const renderModelOption = (option: typeof MODEL_OPTIONS[0], index: number) => (
    <TouchableOpacity 
      key={index}
      style={[
        styles.modelButton,
        selectedModelIndex === index ? styles.selectedModelButton : null
      ]}
      onPress={() => setSelectedModelIndex(index)}
      disabled={modelLoading}
    >
      <View style={styles.modelButtonContent}>
        <Text 
          style={[
            styles.modelButtonText,
            selectedModelIndex === index ? styles.selectedModelButtonText : null
          ]}
        >
          {option.label}
        </Text>
        {option.description && (
          <Text style={styles.modelDescription}>{option.description}</Text>
        )}
      </View>
    </TouchableOpacity>
  );

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Select Embedding Model</Text>
      
      <View style={styles.modelSelector}>
        <View style={styles.modelButtonsContainer}>
          {/* Standard models (always visible) */}
          {renderModelOption(MODEL_OPTIONS[0], 0)}
          {renderModelOption(MODEL_OPTIONS[1], 1)}
          
          {/* Expandable optimized models section */}
          <TouchableOpacity 
            style={styles.sectionToggle}
            onPress={() => setShowOptimizedModels(!showOptimizedModels)}
            disabled={modelLoading}
          >
            <Text style={styles.sectionToggleText}>
              {showOptimizedModels ? '▼ ' : '▶ '}Optimized Variants ({MODEL_OPTIONS.slice(2, 6).length})
            </Text>
          </TouchableOpacity>
          
          {showOptimizedModels && (
            <View style={styles.expandableSection}>
              {MODEL_OPTIONS.slice(2, 6).map((option, i) => 
                renderModelOption(option, i + 2)
              )}
            </View>
          )}
          
          {/* Expandable quantized models section */}
          <TouchableOpacity 
            style={styles.sectionToggle}
            onPress={() => setShowQuantizedModels(!showQuantizedModels)}
            disabled={modelLoading}
          >
            <Text style={styles.sectionToggleText}>
              {showQuantizedModels ? '▼ ' : '▶ '}Quantized Models ({MODEL_OPTIONS.slice(6).length})
            </Text>
          </TouchableOpacity>
          
          {showQuantizedModels && (
            <View style={styles.expandableSection}>
              {MODEL_OPTIONS.slice(6).map((option, i) => 
                renderModelOption(option, i + 6)
              )}
            </View>
          )}
          
          {/* Model download confirmation button */}
          <View style={styles.downloadButtonContainer}>
            <Button
              title={modelLoading 
                ? "Loading..." 
                : modelInfo.readyToUse 
                  ? "Continue to Embedder" 
                  : modelInfo.cached 
                    ? "Load Cached Model" 
                    : "Download & Load Model"
              }
              onPress={modelInfo.readyToUse ? () => onModelReady() : confirmModelDownload}
              disabled={modelLoading || !modelSelected}
              color={modelInfo.readyToUse ? "#4CAF50" : "#007AFF"}
            />
            
            {onBack && (
              <TouchableOpacity 
                style={styles.backButton}
                onPress={onBack}
                disabled={modelLoading}
              >
                <Text style={styles.backButtonText}>← Back</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      </View>
      
      {/* Display model info when available */}
      {modelInfo.size && (
        <View style={[styles.modelInfoContainer, modelInfo.readyToUse ? styles.modelReadyContainer : null]}>
          <Text style={styles.infoLabel}>Model: {MODEL_OPTIONS[selectedModelIndex].label}</Text>
          <Text>Size: {modelInfo.size}</Text>
          {modelInfo.cached && <Text>Status: {modelInfo.readyToUse ? "Ready to use" : "Cached (needs loading)"}</Text>}
          {modelInfo.loadTime && <Text>Load time: {modelInfo.loadTime}ms</Text>}
        </View>
      )}
      
      {(modelLoading) && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#0000ff" />
          <Text>{"Loading model and tokenizer..."}</Text>
        </View>
      )}
      
      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}
    </ScrollView>
  );
};

// Embedding Screen component
const EmbeddingScreen = ({ modelData, onBack, onActions }) => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [embedding, setEmbedding] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showFullEmbedding, setShowFullEmbedding] = useState(false);
  
  const generateEmbedding = async () => {
    if (!modelData.session || !modelData.tokenizer) {
      setError('Model or tokenizer not loaded');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setEmbedding(null);
      
      console.log('Tokenizing input text:', text);
      
      // Use our custom tokenizer implementation with the model-specific padding length
      const tokenized = await modelData.tokenizer.tokenizer(text, {
        padding: true,
        truncation: true,
        max_length: modelData.modelInfo.paddingLength
      });
      
      console.log('Tokenization complete');
      
      // Convert the tokenizer output to ONNX tensors
      const inputIds = new ort.Tensor(
        'int64', 
        Array.from(tokenized.input_ids.data).map(n => BigInt(n)),
        tokenized.input_ids.dims
      );
      
      const attentionMask = new ort.Tensor(
        'int64',
        Array.from(tokenized.attention_mask.data).map(n => BigInt(n)),
        tokenized.attention_mask.dims
      );
      
      // Prepare feeds
      const feeds: { [key: string]: ort.Tensor } = { 
        input_ids: inputIds,
        attention_mask: attentionMask
      };
      
      // Add token_type_ids if needed by the model
      if (tokenized.token_type_ids) {
        feeds.token_type_ids = new ort.Tensor(
          'int64',
          Array.from(tokenized.token_type_ids.data).map(n => BigInt(n)),
          tokenized.token_type_ids.dims
        );
      }
      
      console.log('Running inference with model...');
      const inferenceStartTime = Date.now();
      
      const results = await modelData.session.run(feeds);
      
      const inferenceDuration = Date.now() - inferenceStartTime;
      console.log(`Inference completed in ${inferenceDuration}ms`);
      
      // Try different output keys depending on the model
      const outputData = (results.embeddings as { data: number[] })?.data 
        || (results.output as { data: number[] })?.data 
        || (results.last_hidden_state as { data: number[] })?.data
        || (results.pooler_output as { data: number[] })?.data
        || (results.sentence_embedding as { data: number[] })?.data
        || (Object.values(results)[0] as { data: number[] })?.data;
      
      if (!outputData) {
        throw new Error('No output data found in model results');
      }

      // Get the first output tensor's shape
      const outputTensor = Object.values(results)[0];
      console.log(`Raw output shape: ${outputTensor.dims.join('×')}`);
      
      let finalEmbedding;
      
      // Check if we need to get the CLS token embedding from the output tensor
      if (outputTensor.dims.length === 3) {  // Shape is [batch_size, sequence_length, hidden_size]
        const batchSize = outputTensor.dims[0];
        const seqLength = outputTensor.dims[1];
        const hiddenSize = outputTensor.dims[2];
        
        console.log(`Detected 3D tensor with shape [${batchSize}, ${seqLength}, ${hiddenSize}]`);
        console.log(`Using CLS token (first token) for a ${hiddenSize}-dimensional embedding`);
        
        // Use only the CLS token embedding (first token)
        finalEmbedding = new Array(hiddenSize);
        for (let j = 0; j < hiddenSize; j++) {
          // CLS token is the first token (index 0)
          // Calculate index in the flattened array: (batch_idx * seq_len * hidden + seq_idx * hidden + hidden_idx)
          const idx = (0 * seqLength * hiddenSize) + (0 * hiddenSize) + j;
          finalEmbedding[j] = outputData[idx];
        }
        
        console.log(`Final embedding dimensions (CLS token): ${finalEmbedding.length}`);
      } else {
        // If already a vector, use as is
        finalEmbedding = Array.from(outputData);
        console.log(`Using raw embedding with dimensions: ${finalEmbedding.length}`);
      }
      
      // Update the model info with inference time
      modelData.modelInfo.inferenceTime = inferenceDuration;
      
      setEmbedding(finalEmbedding);
      setLoading(false);
    } catch (err) {
      console.error('Embedding generation error:', err);
      setError('Error generating embedding: ' + (err instanceof Error ? err.message : 'An unknown error occurred'));
      setLoading(false);
    }
  };

  const renderEmbeddingVisualization = () => {
    if (!embedding) return null;
    
    // For full embedding display
    if (showFullEmbedding) {
      return (
        <ScrollView style={styles.fullEmbeddingContainer}>
          <Text style={styles.embedPreview}>
            [{embedding.map(n => n.toString()).join(',\n')}]
          </Text>
          <TouchableOpacity 
            style={styles.toggleViewButton}
            onPress={() => setShowFullEmbedding(false)}
          >
            <Text style={styles.toggleViewButtonText}>Show Summary View</Text>
          </TouchableOpacity>
        </ScrollView>
      );
    }
    
    // For summary embedding display
    return (
      <View>
        <View style={styles.embeddingVisualizer}>
          <Text style={styles.debugText}>Embedding dimensions: {embedding.length}</Text>
          <Text style={styles.debugText}>Sample dimensions: {SAMPLE_EMBEDDING.length}</Text>
          
          {embedding.slice(0, 50).map((value, index) => (
            <View 
              key={index} 
              style={[
                styles.embeddingBar,
                { 
                  height: Math.max(2, Math.abs(value) * 50),
                  backgroundColor: value >= 0 ? '#4CAF50' : '#F44336'
                }
              ]} 
            />
          ))}
        </View>
        
        <View style={styles.embeddingActions}>
          <TouchableOpacity 
            style={styles.toggleViewButton}
            onPress={() => setShowFullEmbedding(true)}
          >
            <Text style={styles.toggleViewButtonText}>Show Full Embedding</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={onBack}
          disabled={loading}
        >
          <Text style={styles.backButtonText}>← Change Model</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Text Embedder</Text>
      </View>
      
      <View style={styles.modelInfoCard}>
        <Text style={styles.infoLabel}>Active Model: {modelData.modelInfo.name}</Text>
        <Text>{modelData.modelInfo.description}</Text>
        {modelData.modelInfo.inferenceTime && (
          <Text>Last inference: {modelData.modelInfo.inferenceTime}ms</Text>
        )}
      </View>
      
      <TextInput
        style={styles.input}
        value={text}
        onChangeText={setText}
        placeholder="Enter text to embed"
        multiline
      />
      
      <Button
        title={loading ? "Generating..." : "Generate Embedding"}
        onPress={generateEmbedding}
        disabled={loading || !text}
        color="#007AFF"
      />
      
      <Button
        title="Debug: Compare with 'hello'"
        onPress={async () => {
          const savedEmbedding = embedding;
          const savedText = text;
          
          // Store current embedding
          if (!savedEmbedding) {
            Alert.alert("Error", "Generate an embedding first");
            return;
          }
          
          // Generate embedding for "hello"
          setText("hello");
          await generateEmbedding();
          
          // Compare embeddings
          const helloEmbedding = embedding;
          const similarity = calculateCosineSimilarity(savedEmbedding, helloEmbedding);
          
          // Reset state and show result
          setText(savedText);
          setEmbedding(savedEmbedding);
          
          Alert.alert(
            "Embedding Comparison", 
            `Similarity between "${savedText}" and "hello": ${(similarity * 100).toFixed(2)}%\n\n` +
            `First 5 values of "${savedText}":\n${savedEmbedding.slice(0, 5).map(n => n.toFixed(5)).join(', ')}\n\n` +
            `First 5 values of "hello":\n${helloEmbedding.slice(0, 5).map(n => n.toFixed(5)).join(', ')}`
          );
        }}
        color="#FF9800"
      />
      
      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#0000ff" />
          <Text>Generating embedding...</Text>
        </View>
      )}
      
      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}
      
      {embedding && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>Embedding generated!</Text>
          <Text>Dimensions: {embedding.length}</Text>
          <Text style={styles.embedPreviewHeader}>
            Preview: [{embedding.slice(0, 5).map(n => n.toString()).join(', ')}, ...]
          </Text>
          
          {renderEmbeddingVisualization()}
          
          {/* New button to go to actions screen */}
          <TouchableOpacity 
            style={styles.actionButton}
            onPress={() => onActions(embedding, text)}
          >
            <Text style={styles.actionButtonText}>Store or Compare Embedding</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
};

// Embedding Actions Screen component
const EmbeddingActionsScreen = ({ modelData, embedding, text, onBack, onNewEmbedding }) => {
  const [sampleEmbedding, setSampleEmbedding] = useState(null);
  const [similarity, setSimilarity] = useState(null);
  const [isComparingWithSample, setIsComparingWithSample] = useState(false);
  const [precomputedComparison, setPrecomputedComparison] = useState(null);
  const [comparisonStatus, setComparisonStatus] = useState('');
  
  const generateSampleEmbedding = async () => {
    try {
      setIsComparingWithSample(true);
      setSimilarity(null);
      setComparisonStatus('Comparing with new sample...');
      
      console.log('Generating embedding for sample text');
      
      // Tokenize the sample text
      const tokenized = await modelData.tokenizer.tokenizer(SAMPLE_TEXT, {
        padding: true,
        truncation: true,
        max_length: modelData.modelInfo.paddingLength
      });
      
      // Convert to ONNX tensors
      const inputIds = new ort.Tensor(
        'int64', 
        Array.from(tokenized.input_ids.data).map(n => BigInt(n)),
        tokenized.input_ids.dims
      );
      
      const attentionMask = new ort.Tensor(
        'int64',
        Array.from(tokenized.attention_mask.data).map(n => BigInt(n)),
        tokenized.attention_mask.dims
      );
      
      // Prepare feeds
      const feeds = { 
        input_ids: inputIds,
        attention_mask: attentionMask
      };
      
      // Add token_type_ids if needed by the model
      if (tokenized.token_type_ids) {
        feeds.token_type_ids = new ort.Tensor(
          'int64',
          Array.from(tokenized.token_type_ids.data).map(n => BigInt(n)),
          tokenized.token_type_ids.dims
        );
      }
      
      // Run inference
      const results = await modelData.session.run(feeds);
      
      // Get the output data
      const outputData = (results.embeddings as { data: number[] })?.data 
        || (results.output as { data: number[] })?.data 
        || (results.last_hidden_state as { data: number[] })?.data
        || (results.pooler_output as { data: number[] })?.data
        || (results.sentence_embedding as { data: number[] })?.data
        || (Object.values(results)[0] as { data: number[] })?.data;
      
      if (!outputData) {
        throw new Error('No output data found in sample results');
      }
      
      const sampleEmbeddingArray = Array.from(outputData);
      setSampleEmbedding(sampleEmbeddingArray);
      
      // Calculate similarity
      const similarityScore = calculateCosineSimilarity(embedding, sampleEmbeddingArray);
      setSimilarity(similarityScore);
      
      setIsComparingWithSample(false);
      setComparisonStatus('');
    } catch (error) {
      console.error('Error generating sample embedding:', error);
      setComparisonStatus(`Error in comparison: ${error.message}`);
      setIsComparingWithSample(false);
    }
  };
  
  // Add a new function to compare with the pre-computed sample
  const compareWithPrecomputed = () => {
    setComparisonStatus('Comparing with pre-computed sample...');
    const result = compareWithPrecomputedSample(embedding);
    setPrecomputedComparison(result);
    setComparisonStatus('');
  };
  
  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={onBack}
        >
          <Text style={styles.backButtonText}>← Back to Embedding</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Embedding Comparison</Text>
      </View>

      <View style={styles.modelInfoCard}>
        <Text style={styles.infoLabel}>Active Model: {modelData.modelInfo.name}</Text>
        <Text>Embedding Size: {embedding.length} dimensions</Text>
        <Text style={styles.embeddingSourceText}>Source Text: "{text.length > 50 ? text.substring(0, 47) + '...' : text}"</Text>
      </View>

      {/* Action buttons */}
      <View style={styles.actionButtonsContainer}>
        <TouchableOpacity 
          style={styles.actionButton}
          onPress={generateSampleEmbedding}
          disabled={isComparingWithSample}
        >
          <Text style={styles.actionButtonText}>
            {isComparingWithSample ? 'Comparing...' : 'Generate & Compare with Sample'}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.actionButton, { backgroundColor: '#4CAF50' }]}
          onPress={compareWithPrecomputed}
        >
          <Text style={styles.actionButtonText}>
            Compare with Malaysian Politics Query
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.actionButton, { backgroundColor: '#FF9800' }]}
          onPress={onNewEmbedding}
        >
          <Text style={styles.actionButtonText}>Generate New Embedding</Text>
        </TouchableOpacity>
      </View>
      
      {/* Status message */}
      {comparisonStatus ? (
        <View style={styles.statusContainer}>
          <Text style={styles.statusText}>{comparisonStatus}</Text>
        </View>
      ) : null}
      
      {/* Sample comparison results */}
      {isComparingWithSample && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#0000ff" />
          <Text>Comparing with sample text...</Text>
        </View>
      )}
      
      {similarity !== null && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>Comparison Results</Text>
          <Text style={styles.comparisonLabel}>Sample Text:</Text>
          <Text style={styles.comparisonText}>{SAMPLE_TEXT}</Text>
          <Text style={styles.comparisonLabel}>Cosine Similarity:</Text>
          <Text style={styles.similarityScore}>
            {(similarity * 100).toFixed(2)}%
          </Text>
          <View style={[
            styles.similarityBar, 
            { 
              width: `${Math.max(1, similarity * 100)}%`,
              backgroundColor: similarity > 0.7 ? '#4CAF50' : 
                              similarity > 0.4 ? '#FFC107' : '#F44336'
            }
          ]} />
          <Text style={styles.comparisonNote}>
            {similarity > 0.8 ? 'Very Similar' : 
             similarity > 0.5 ? 'Moderately Similar' : 
             similarity > 0.3 ? 'Slightly Similar' : 'Not Similar'}
          </Text>
        </View>
      )}
      
      {/* Display results for pre-computed comparison */}
      {precomputedComparison && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>Pre-computed Comparison Results</Text>
          <Text style={styles.comparisonLabel}>Pre-computed Sample:</Text>
          <Text style={styles.comparisonText}>{SAMPLE_TEXT}</Text>
          <Text style={styles.comparisonLabel}>Cosine Similarity:</Text>
          <Text style={styles.similarityScore}>
            {(precomputedComparison.similarity * 100).toFixed(2)}%
          </Text>
          <View style={[
            styles.similarityBar, 
            { 
              width: `${Math.max(1, precomputedComparison.similarity * 100)}%`,
              backgroundColor: precomputedComparison.similarity > 0.5 ? '#4CAF50' : '#FFA726'
            }
          ]} />
          <Text style={styles.comparisonNote}>
            {precomputedComparison.message}
          </Text>
        </View>
      )}
    </ScrollView>
  );
};

// Helper functions
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const createTokenizer = (tokenizerConfig: any, tokenizerData: any, maxPadding: number, needsTokenTypeIds: boolean) => {
  // Word piece tokenization helper
  const wordPieceTokenize = (text: string) => {
    // Basic implementation of WordPiece tokenization
    const tokens: string[] = [];
    const words = text.split(/\s+/);
    
    for (const word of words) {
      if (!word) continue;
      
      // Check if the whole word is in vocabulary
      if (tokenizerData.model.vocab[word] !== undefined) {
        tokens.push(word);
        continue;
      }
      
      // Try to split the word into subwords
      let subTokens: string[] = [];
      let subWord = '';
      let remainingChars = word;
      
      // Try to find the longest matching subword
      while (remainingChars.length > 0) {
        let found = false;
        for (let endPos = remainingChars.length; endPos > 0; endPos--) {
          const candidate = remainingChars.substring(0, endPos);
          if (tokenizerData.model.vocab[candidate] !== undefined) {
            subTokens.push(candidate);
            remainingChars = remainingChars.substring(endPos);
            found = true;
            break;
          }
        }
        
        // If no subword was found, use UNK and skip this character
        if (!found) {
          subTokens.push(tokenizerConfig.unk_token || "[UNK]");
          remainingChars = remainingChars.substring(1);
        }
      }
      
      tokens.push(...subTokens);
    }
    
    return tokens;
  };

  // Convert tokens to token IDs using the vocab from tokenizer data
  const convertTokensToIds = (tokens: string[]) => {
    return tokens.map(token => {
      const id = tokenizerData.model.vocab[token];
      return id !== undefined ? id : tokenizerData.model.vocab[tokenizerConfig.unk_token || "[UNK]"] || 0;
    });
  };
  
  return {
    config: tokenizerConfig,
    data: tokenizerData,
    needsTokenTypeIds,
    
    tokenizer: async (inputText: string, options: any = {}) => {
      // Process options
      const maxLength = options.max_length || maxPadding;
      const padding = options.padding !== false;
      
      // Add special tokens and clean text according to model needs
      const cleanedText = tokenizerConfig.do_lower_case ? inputText.toLowerCase() : inputText;
      
      // Tokenize
      const tokens = wordPieceTokenize(cleanedText);
      
      // Get special tokens from config
      const clsToken = tokenizerConfig.cls_token || "[CLS]";
      const sepToken = tokenizerConfig.sep_token || "[SEP]";
      const padToken = tokenizerConfig.pad_token || "[PAD]";
      
      // For the fixed-length model, we need a specific format:
      // [CLS] + (tokens truncated to fit) + [SEP] + padding to maxLength
      
      // Calculate how many regular tokens we can include
      const specialTokenCount = 2; // [CLS] and [SEP]
      const maxRegularTokens = maxLength - specialTokenCount;
      const truncatedTokens = tokens.slice(0, maxRegularTokens);
      
      // Build the token sequence with exactly maxLength tokens
      let processedTokens: string[] = [];
      processedTokens.push(clsToken);
      processedTokens = [...processedTokens, ...truncatedTokens];
      processedTokens.push(sepToken);
      
      // Add padding if needed
      if (padding && processedTokens.length < maxLength) {
        const padTokens = new Array(maxLength - processedTokens.length).fill(padToken);
        processedTokens = [...processedTokens, ...padTokens];
      }
      
      // Convert to IDs
      const inputIds = convertTokensToIds(processedTokens);
      
      // Create attention mask (1 for real tokens, 0 for padding)
      const attentionMask = processedTokens.map(token => token === padToken ? 0 : 1);
      
      // Create token type IDs (all zeros for single sentence)
      const tokenTypeIds = new Array(processedTokens.length).fill(0);
      
      console.log(`Tokenized "${inputText}" to ${inputIds.length} tokens with padding to ${maxLength}`);
      
      // Format in the expected shape for ONNX
      const result: any = {
        input_ids: {
          data: new Int32Array(inputIds),
          dims: [1, inputIds.length]
        },
        attention_mask: {
          data: new Int32Array(attentionMask),
          dims: [1, attentionMask.length]
        }
      };
      
      // Add token_type_ids if needed by model
      if (needsTokenTypeIds) {
        result.token_type_ids = {
          data: new Int32Array(tokenTypeIds),
          dims: [1, tokenTypeIds.length]
        };
      }
      
      return result;
    }
  };
};

// Main component with screen navigation
export const NativeTextEmbedder = () => {
  const [currentScreen, setCurrentScreen] = useState('modelSelection');
  const [modelData, setModelData] = useState(null);
  const [currentEmbedding, setCurrentEmbedding] = useState(null);
  const [currentText, setCurrentText] = useState('');
  
  const handleModelReady = (data) => {
    setModelData(data);
    setCurrentScreen('embedding');
  };
  
  const goToModelSelection = () => {
    setCurrentScreen('modelSelection');
  };
  
  const goToEmbeddingActions = (embedding, text) => {
    setCurrentEmbedding(embedding);
    setCurrentText(text);
    setCurrentScreen('embeddingActions');
  };
  
  const goToEmbedding = () => {
    setCurrentScreen('embedding');
  };
  
  return (
    <View style={styles.mainContainer}>
      {currentScreen === 'modelSelection' ? (
        <ModelSelectionScreen 
          onModelReady={handleModelReady}
        />
      ) : currentScreen === 'embedding' ? (
        <EmbeddingScreen 
          modelData={modelData}
          onBack={goToModelSelection}
          onActions={goToEmbeddingActions}
        />
      ) : (
        <EmbeddingActionsScreen
          modelData={modelData}
          embedding={currentEmbedding}
          text={currentText}
          onBack={goToEmbedding}
          onNewEmbedding={goToEmbedding}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  mainContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
    flex: 1
  },
  modelSelector: {
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 10,
  },
  label: {
    fontSize: 16,
    marginBottom: 10,
    fontWeight: 'bold',
  },
  modelButtonsContainer: {
    flexDirection: 'column',
    gap: 8,
  },
  modelButton: {
    padding: 10,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    backgroundColor: '#f5f5f5',
  },
  modelButtonContent: {
    flexDirection: 'column',
  },
  selectedModelButton: {
    borderColor: '#007AFF',
    backgroundColor: '#e6f2ff',
  },
  modelButtonText: {
    fontSize: 14,
    color: '#333',
  },
  modelDescription: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  selectedModelButtonText: {
    fontWeight: 'bold',
    color: '#007AFF',
  },
  sectionToggle: {
    padding: 10,
    marginVertical: 5,
    backgroundColor: '#f0f0f0',
    borderRadius: 5,
    borderLeftWidth: 3,
    borderLeftColor: '#007AFF',
  },
  sectionToggleText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  expandableSection: {
    marginLeft: 10,
    borderLeftWidth: 1,
    borderLeftColor: '#ccc',
    paddingLeft: 10,
  },
  downloadButtonContainer: {
    marginTop: 15,
    padding: 10,
    backgroundColor: '#f8f8f8',
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#ddd',
    borderStyle: 'dashed',
  },
  modelInfoContainer: {
    marginBottom: 15,
    padding: 10,
    backgroundColor: '#f8f8f8',
    borderRadius: 5,
    borderLeftWidth: 4,
    borderLeftColor: '#007AFF',
  },
  modelReadyContainer: {
    borderLeftColor: '#4CAF50',
    backgroundColor: '#f1f8e9',
  },
  modelInfoCard: {
    marginBottom: 15,
    padding: 10,
    backgroundColor: '#e6f2ff',
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  infoLabel: {
    fontWeight: 'bold',
    marginBottom: 5,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 10,
    minHeight: 100,
    marginBottom: 20,
  },
  loadingContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  errorContainer: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#ffeeee',
    borderRadius: 5,
  },
  errorText: {
    color: 'red',
  },
  resultContainer: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#eeffee',
    borderRadius: 5,
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  embedPreviewHeader: {
    fontFamily: 'monospace',
    marginTop: 10,
    marginBottom: 15,
  },
  embedPreview: {
    fontFamily: 'monospace',
    marginTop: 10,
  },
  backButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 5,
    backgroundColor: '#f0f0f0',
    marginRight: 10,
  },
  backButtonText: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
  embeddingVisualizer: {
    flexDirection: 'row',
    height: 100,
    marginVertical: 15,
    alignItems: 'flex-end',
    borderBottomWidth: 1,
    borderColor: '#ccc',
    paddingBottom: 5,
  },
  embeddingBar: {
    flex: 1,
    margin: 1,
    minWidth: 2,
  },
  embeddingActions: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 10,
  },
  toggleViewButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 5,
    marginTop: 10,
  },
  toggleViewButtonText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  fullEmbeddingContainer: {
    maxHeight: 300,
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f5f5f5',
    borderRadius: 5,
  },
  actionButtonsContainer: {
    marginVertical: 15,
    gap: 10,
  },
  actionButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    marginTop: 15,
    alignItems: 'center',
  },
  actionButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  statusContainer: {
    marginVertical: 15,
    padding: 10,
    backgroundColor: '#f0f8ff',
    borderRadius: 5,
    borderLeftWidth: 4,
    borderLeftColor: '#007AFF',
  },
  statusText: {
    color: '#333',
  },
  storedEmbeddingsContainer: {
    marginTop: 20,
    borderTopWidth: 1,
    borderTopColor: '#ddd',
    paddingTop: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  embeddingListItem: {
    padding: 12,
    backgroundColor: '#f5f5f5',
    borderRadius: 6,
    marginBottom: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#007AFF',
  },
  embeddingItemText: {
    fontSize: 14,
  },
  embeddingItemMeta: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  emptyListText: {
    fontStyle: 'italic',
    color: '#666',
    textAlign: 'center',
    padding: 20,
  },
  comparisonLabel: {
    fontWeight: 'bold',
    marginTop: 8,
  },
  comparisonText: {
    fontStyle: 'italic',
    marginVertical: 8,
    padding: 8,
    backgroundColor: '#f5f5f5',
    borderRadius: 4,
  },
  similarityScore: {
    fontSize: 24,
    fontWeight: 'bold',
    marginVertical: 8,
    color: '#007AFF',
  },
  similarityBar: {
    height: 10,
    backgroundColor: '#4CAF50',
    borderRadius: 5,
    marginVertical: 8,
  },
  comparisonNote: {
    fontStyle: 'italic',
    color: '#666',
  },
  embeddingSourceText: {
    fontStyle: 'italic',
    marginTop: 5,
    color: '#555',
  },
  debugText: {
    color: '#666',
    fontSize: 12,
    marginBottom: 5,
    fontFamily: 'monospace',
  },
});