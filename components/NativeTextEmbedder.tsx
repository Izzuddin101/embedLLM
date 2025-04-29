import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ActivityIndicator, ScrollView, TouchableOpacity, Platform, Alert } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ort from 'onnxruntime-react-native';

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
const EmbeddingScreen = ({ modelData, onBack }) => {
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
      
      // Update the model info with inference time
      modelData.modelInfo.inferenceTime = inferenceDuration;
      
      setEmbedding(Array.from(outputData));
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
            [{embedding.map(n => n.toFixed(4)).join(',\n')}]
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
            Preview: [{embedding.slice(0, 5).map(n => n.toFixed(4)).join(', ')}, ...]
          </Text>
          
          {renderEmbeddingVisualization()}
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
      
      for (let i = 0; i < word.length; i++) {
        subWord += word[i];
        
        // Check if current subword is in vocabulary
        if (tokenizerData.model.vocab[subWord] !== undefined) {
          subTokens.push(subWord);
          subWord = '';
        }
        // If we've reached the end and couldn't tokenize
        else if (i === word.length - 1) {
          // Use UNK token
          subTokens.push(tokenizerConfig.unk_token || "[UNK]");
          break;
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
  
  const handleModelReady = (data) => {
    setModelData(data);
    setCurrentScreen('embedding');
  };
  
  const goToModelSelection = () => {
    setCurrentScreen('modelSelection');
  };
  
  return (
    <View style={styles.mainContainer}>
      {currentScreen === 'modelSelection' ? (
        <ModelSelectionScreen 
          onModelReady={handleModelReady}
        />
      ) : (
        <EmbeddingScreen 
          modelData={modelData}
          onBack={goToModelSelection}
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
  }
});