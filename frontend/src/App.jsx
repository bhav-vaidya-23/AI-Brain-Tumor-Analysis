import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  UploadCloud, 
  FileImage, 
  ShieldCheck, 
  AlertTriangle, 
  CheckCircle2, 
  Layers, 
  Eye, 
  Cpu, 
  RefreshCw, 
  Info, 
  ChevronRight, 
  Sparkles, 
  Image as ImageIcon,
  Check,
  AlertCircle,
  BarChart3,
  HelpCircle,
  Download
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const DEMO_SAMPLES = [
  { id: 'glioma', label: 'Glioma Sample', path: '/samples/glioma_sample.jpg', hint: 'Frontal lobe infiltration' },
  { id: 'meningioma', label: 'Meningioma Sample', path: '/samples/meningioma_sample.jpg', hint: 'Dural-based extra-axial lesion' },
  { id: 'pituitary', label: 'Pituitary Sample', path: '/samples/pituitary_sample.jpg', hint: 'Sellar/parasellar lesion' },
  { id: 'no_tumor', label: 'No Tumor Sample', path: '/samples/no_tumor_sample.jpg', hint: 'Normal anatomical MRI' },
];

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [activeTab, setActiveTab] = useState('all');
  const [selectedDemo, setSelectedDemo] = useState(null);
  const fileInputRef = useRef(null);

  // Loading progression messages
  const loadingSteps = [
    'Decoding MRI slice in memory...',
    'Extracting ResNet-18 deep features...',
    'Computing layer4 Grad-CAM explainability saliency...',
    'Delineating U-Net tumor segmentation boundary (Epoch 23)...',
    'Synthesizing multi-modal overlays and confidence scores...'
  ];

  // Check backend health on mount
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth({ status: 'offline' });
      }
    } catch {
      setHealth({ status: 'offline' });
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(selected.type)) {
      setError('Please upload a valid MRI image in JPG or PNG format.');
      return;
    }
    if (selected.size > 10 * 1024 * 1024) {
      setError('File size exceeds the 10MB limit.');
      return;
    }

    setError(null);
    setSelectedDemo(null);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResult(null);
  };

  const handleSelectDemo = async (demo) => {
    setError(null);
    setSelectedDemo(demo.id);
    setResult(null);

    try {
      const res = await fetch(demo.path);
      const blob = await res.blob();
      const demoFile = new File([blob], `${demo.id}_sample.jpg`, { type: 'image/jpeg' });
      setFile(demoFile);
      setPreviewUrl(demo.path);
    } catch (err) {
      setError('Failed to load demo sample image.');
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setLoadingStep(0);

    // Simulate multi-step progress bar
    const stepInterval = setInterval(() => {
      setLoadingStep((prev) => (prev < loadingSteps.length - 1 ? prev + 1 : prev));
    }, 600);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData
      });

      clearInterval(stepInterval);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      setActiveTab('all');
    } catch (err) {
      clearInterval(stepInterval);
      setError(err.message || 'An error occurred during analysis. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setSelectedDemo(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30">
      
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Activity className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-white font-['Outfit']">NeuroVision AI</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  Research Prototype
                </span>
              </div>
              <p className="text-xs text-slate-400">Classification • Grad-CAM Explainability • U-Net Segmentation</p>
            </div>
          </div>

          {/* Backend Health Badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-900 border border-slate-800">
              <span className="relative flex h-2 w-2">
                {health?.status === 'healthy' ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                )}
              </span>
              <span className={health?.status === 'healthy' ? 'text-slate-300' : 'text-rose-400'}>
                {health?.status === 'healthy' ? 'Engine Ready' : 'Engine Offline'}
              </span>
            </div>
          </div>

        </div>
      </header>

      {/* Mandatory Medical Disclaimer Banner */}
      <div className="bg-amber-950/40 border-b border-amber-500/20 px-4 py-2.5">
        <div className="max-w-7xl mx-auto flex items-center gap-3 text-amber-200/90 text-xs sm:text-sm">
          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
          <p>
            <strong className="font-semibold text-amber-300">Investigational Prototype:</strong> This application is an AI-assisted research system and is <u>not</u> clinically validated. Outputs must not be used for diagnosis, medical decision-making, or surgical guidance.
          </p>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        {/* Intro Hero Section (When no result) */}
        {!result && (
          <div className="text-center max-w-3xl mx-auto space-y-3 pt-2">
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white font-['Outfit'] tracking-tight">
              Explainable Brain Tumor MRI Analysis
            </h1>
            <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
              Upload a T1-weighted brain MRI scan to run ResNet-18 multi-class identification, layer4 Grad-CAM visual attribution maps, and 4-level U-Net pixel segmentation in real-time.
            </p>
          </div>
        )}

        {/* Error Alert Box */}
        {error && (
          <div className="max-w-3xl mx-auto p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 flex items-start gap-3 text-rose-200 text-sm animate-fade-in">
            <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold text-rose-300">Analysis Request Failed</p>
              <p className="text-rose-300/80 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Upload & Workspace Section */}
        {!result && (
          <div className="max-w-3xl mx-auto space-y-6">
            
            {/* Upload Dropzone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
                previewUrl 
                  ? 'border-cyan-500/40 bg-slate-900/40' 
                  : 'border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/30 bg-slate-900/20'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                onChange={handleFileChange}
                className="hidden"
              />

              {previewUrl ? (
                <div className="space-y-4">
                  <div className="relative inline-block rounded-xl overflow-hidden border border-slate-700 bg-black/60 shadow-xl">
                    <img 
                      src={previewUrl} 
                      alt="Selected MRI Preview" 
                      className="h-56 w-56 object-cover object-center"
                    />
                    <div className="absolute top-2 right-2 bg-slate-950/80 px-2 py-0.5 rounded text-[10px] font-mono text-cyan-400 border border-cyan-500/30">
                      256 × 256
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-200">{file?.name}</p>
                    <p className="text-xs text-slate-500 mt-0.5">Click to change or drag another image</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4 py-4">
                  <div className="h-16 w-16 mx-auto rounded-2xl bg-gradient-to-tr from-cyan-500/10 to-indigo-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 shadow-inner">
                    <UploadCloud className="h-8 w-8" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-base font-semibold text-slate-200">
                      Click to upload or drag and drop MRI slice
                    </p>
                    <p className="text-xs text-slate-500">
                      Supported formats: JPG, JPEG, PNG (T1 Axial/Coronal/Sagittal) • Max 10MB
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Quick Demo Samples */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-medium">Or select a preloaded sample scan:</span>
                <span className="text-[11px] text-slate-500">Instant 1-click test</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {DEMO_SAMPLES.map((demo) => (
                  <button
                    key={demo.id}
                    onClick={() => handleSelectDemo(demo)}
                    className={`p-3 rounded-xl text-left border transition-all ${
                      selectedDemo === demo.id
                        ? 'bg-cyan-950/40 border-cyan-500 text-cyan-200 ring-1 ring-cyan-500'
                        : 'bg-slate-900/50 border-slate-800 text-slate-300 hover:bg-slate-800/60 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold">{demo.label}</span>
                      {selectedDemo === demo.id && <Check className="h-3.5 w-3.5 text-cyan-400" />}
                    </div>
                    <span className="text-[10px] text-slate-500 block mt-1 line-clamp-1">{demo.hint}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Analyze Action Button */}
            <div className="pt-2">
              <button
                disabled={!file || loading}
                onClick={handleAnalyze}
                className={`w-full py-4 rounded-xl font-semibold text-base flex items-center justify-center gap-2 shadow-lg transition-all ${
                  !file || loading
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-cyan-500 via-indigo-600 to-cyan-500 bg-size-200 hover:bg-pos-100 text-white shadow-cyan-500/20 hover:shadow-cyan-500/30'
                }`}
              >
                {loading ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Processing In-Memory Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5" />
                    <span>Run Integrated AI Analysis</span>
                  </>
                )}
              </button>
            </div>

            {/* Loading Progression Card */}
            {loading && (
              <div className="glass-panel p-6 rounded-2xl border border-cyan-500/30 space-y-4 animate-pulse">
                <div className="flex items-center justify-between text-xs text-cyan-400 font-mono">
                  <span>Executing Pipeline</span>
                  <span>Step {loadingStep + 1} of {loadingSteps.length}</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-cyan-400 to-indigo-500 h-full transition-all duration-300"
                    style={{ width: `${((loadingStep + 1) / loadingSteps.length) * 100}%` }}
                  ></div>
                </div>
                <p className="text-sm text-slate-300 flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-cyan-400 animate-spin" />
                  {loadingSteps[loadingStep]}
                </p>
              </div>
            )}

          </div>
        )}

        {/* Results Dashboard Section */}
        {result && (
          <div className="space-y-8 animate-fade-in">
            
            {/* Action Bar */}
            <div className="flex flex-wrap items-center justify-between gap-4 pb-2 border-b border-slate-800">
              <div>
                <h2 className="text-2xl font-bold text-white font-['Outfit']">Analysis Summary</h2>
                <p className="text-xs text-slate-400">Synthesized inference from ResNet-18, Grad-CAM (layer4), and U-Net (Epoch 23)</p>
              </div>
              <button
                onClick={handleReset}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 flex items-center gap-2 transition"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>Analyze Another Scan</span>
              </button>
            </div>

            {/* Top Row: AI Category + Class Probabilities + Segmentation Insight */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Card 1: Primary AI Prediction */}
              <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 space-y-4 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-2xl -mr-10 -mt-10 pointer-events-none"></div>
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold uppercase tracking-wider">AI-Predicted Category</span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">ResNet-18</span>
                </div>

                <div>
                  <div className="text-3xl font-extrabold text-white font-['Outfit'] tracking-tight">
                    {result.prediction.class}
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-sm font-semibold text-cyan-400">
                      {result.prediction.confidence_percentage}%
                    </span>
                    <span className="text-xs text-slate-500">model confidence score</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-xs">
                  <span className="text-slate-400">AI-Predicted Region:</span>
                  <span className={`font-semibold px-2 py-0.5 rounded ${
                    result.prediction.raw_class === 'no_tumor'
                      ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
                      : result.segmentation.tumor_area_percentage > 0
                        ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20'
                        : 'bg-slate-500/10 text-slate-300 border border-slate-500/20'
                  }`}>
                    {result.prediction.raw_class === 'no_tumor'
                      ? 'Non-lesional (No tumor region)'
                      : result.segmentation.tumor_area_percentage > 0 
                        ? `AI-predicted region (${result.segmentation.tumor_area_percentage}% slice)` 
                        : 'Diffuse / Non-focal'}
                  </span>
                </div>
              </div>

              {/* Card 2: Probability Distribution */}
              <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 space-y-4 lg:col-span-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    <BarChart3 className="h-3.5 w-3.5 text-cyan-400" />
                    Category Probability Distribution
                  </span>
                  <span className="text-[11px] text-slate-500">Softmax Logits</span>
                </div>

                <div className="space-y-3 pt-1">
                  {Object.entries(result.probabilities).map(([clsName, prob]) => {
                    const pct = (prob * 100).toFixed(1);
                    const isTop = clsName === result.prediction.class;

                    return (
                      <div key={clsName} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className={isTop ? 'font-bold text-white' : 'text-slate-400'}>
                            {clsName}
                          </span>
                          <span className={isTop ? 'font-mono font-bold text-cyan-400' : 'font-mono text-slate-500'}>
                            {pct}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              isTop 
                                ? 'bg-gradient-to-r from-cyan-500 to-indigo-500' 
                                : 'bg-slate-700'
                            }`}
                            style={{ width: `${Math.max(parseFloat(pct), 1)}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>

            {/* Visualizations Panel */}
            <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden space-y-6">
              
              {/* Tab Selector */}
              <div className="px-6 pt-6 border-b border-slate-800/80 flex flex-wrap gap-2">
                {[
                  { id: 'all', label: 'Multi-Modal Overview (4 Panels)', icon: Layers },
                  { id: 'gradcam', label: 'Grad-CAM Explainability', icon: Eye },
                  { id: 'unet', label: 'U-Net Segmentation', icon: Activity },
                ].map((tab) => {
                  const IconComp = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`pb-3 px-3 text-xs sm:text-sm font-medium border-b-2 flex items-center gap-2 transition ${
                        activeTab === tab.id
                          ? 'border-cyan-400 text-cyan-300'
                          : 'border-transparent text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <IconComp className="h-4 w-4" />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* Tab Content: All 4 Panels */}
              {activeTab === 'all' && (
                <div className="p-6 pt-0 space-y-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    
                    {/* Panel 1: Original */}
                    <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800 space-y-2">
                      <span className="text-xs font-semibold text-slate-300 block">1. Original T1 MRI</span>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-800">
                        <img src={result.visualizations.original_mri} alt="Original MRI" className="w-full h-full object-cover" />
                      </div>
                      <p className="text-[11px] text-slate-500">Preprocessed to 256×256</p>
                    </div>

                    {/* Panel 2: Grad-CAM Heatmap */}
                    <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800 space-y-2">
                      <span className="text-xs font-semibold text-slate-300 block">2. Grad-CAM Heatmap</span>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-800">
                        <img src={result.visualizations.gradcam_heatmap} alt="Grad-CAM Heatmap" className="w-full h-full object-cover" />
                      </div>
                      <p className="text-[11px] text-slate-500">model.layer4 feature attribution</p>
                    </div>

                    {/* Panel 3: Grad-CAM Overlay */}
                    <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800 space-y-2">
                      <span className="text-xs font-semibold text-slate-300 block">3. Grad-CAM Overlay</span>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-800">
                        <img src={result.visualizations.gradcam_overlay} alt="Grad-CAM Overlay" className="w-full h-full object-cover" />
                      </div>
                      <p className="text-[11px] text-slate-500">Visual attention spatial mapping</p>
                    </div>

                    {/* Panel 4: U-Net Segmentation */}
                    <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800 space-y-2">
                      <span className="text-xs font-semibold text-slate-300 block">4. U-Net Prediction</span>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black border border-slate-800">
                        <img src={result.visualizations.segmentation_overlay} alt="U-Net Overlay" className="w-full h-full object-cover" />
                      </div>
                      <p className="text-[11px] text-slate-500">
                        {result.prediction.raw_class === 'no_tumor' 
                          ? 'Non-lesional slice' 
                          : result.segmentation.tumor_area_percentage > 0 
                            ? `AI-predicted region (${result.segmentation.tumor_area_percentage}% slice)` 
                            : 'No focal region'}
                      </p>
                    </div>

                  </div>
                </div>
              )}

              {/* Tab Content: Grad-CAM Deep Dive */}
              {activeTab === 'gradcam' && (
                <div className="p-6 pt-0 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <span className="text-xs font-semibold text-slate-300">Grad-CAM Overlay on Original MRI</span>
                      <div className="aspect-square rounded-xl overflow-hidden bg-black border border-slate-800 shadow-2xl">
                        <img src={result.visualizations.gradcam_overlay} alt="Grad-CAM Overlay" className="w-full h-full object-cover" />
                      </div>
                    </div>
                    <div className="space-y-4 text-xs sm:text-sm text-slate-300 leading-relaxed bg-slate-900/40 p-6 rounded-xl border border-slate-800 flex flex-col justify-center">
                      <h4 className="text-base font-bold text-white font-['Outfit'] flex items-center gap-2">
                        <Eye className="h-4 w-4 text-cyan-400" />
                        Understanding Grad-CAM Explainability
                      </h4>
                      <p>
                        <strong>Gradient-weighted Class Activation Mapping (Grad-CAM)</strong> computes the gradients of the predicted category score (<code>{result.prediction.class}</code>) with respect to the final convolutional feature maps in <strong>ResNet-18 (layer4)</strong>.
                      </p>
                      <p>
                        Warm regions (red/yellow) indicate anatomical textures and spatial areas that positively influenced the classifier's category assignment.
                      </p>
                      <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-400 space-y-1">
                        <p className="font-semibold text-slate-300">Scientific Interpretation Notice:</p>
                        <p>Grad-CAM represents coarse feature attribution and must not be interpreted as an automated tumor segmentation or clinical surgical margin.</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Tab Content: U-Net Deep Dive */}
              {activeTab === 'unet' && (
                <div className="p-6 pt-0 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <span className="text-xs font-semibold text-slate-300">U-Net Delineation Overlay</span>
                      <div className="aspect-square rounded-xl overflow-hidden bg-black border border-slate-800 shadow-2xl">
                        <img src={result.visualizations.segmentation_overlay} alt="U-Net Overlay" className="w-full h-full object-cover" />
                      </div>
                    </div>
                    <div className="space-y-4 text-xs sm:text-sm text-slate-300 leading-relaxed bg-slate-900/40 p-6 rounded-xl border border-slate-800 flex flex-col justify-center">
                      <h4 className="text-base font-bold text-white font-['Outfit'] flex items-center gap-2">
                        <Activity className="h-4 w-4 text-rose-400" />
                        4-Level U-Net Segmentation Delineation
                      </h4>
                      <p>
                        The 4-level U-Net model predicts pixel-level binary tumor boundaries from single-channel 256×256 normalized MRI slices using an activation threshold of <code>{result.segmentation.threshold_used}</code>.
                      </p>
                      {result.prediction.raw_class === 'no_tumor' ? (
                        <div className="p-3 bg-emerald-950/40 border border-emerald-500/30 rounded-lg text-xs text-emerald-300 space-y-1">
                          <p className="font-semibold">Classification Result: No Tumor</p>
                          <p>The classifier identified this scan as non-lesional. U-Net was trained on tumor cases; therefore, non-lesional scans have no assigned tumor boundaries.</p>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 gap-3 pt-2">
                          <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                            <span className="text-xs text-slate-500 block">AI-Predicted Region Area</span>
                            <span className="text-base font-bold text-white">{result.segmentation.tumor_area_percentage}%</span>
                          </div>
                          <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                            <span className="text-xs text-slate-500 block">Decision Threshold</span>
                            <span className="text-base font-bold text-white">&ge; 0.50</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

            </div>

            {/* Privacy & Technical Footer Notice */}
            <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800/80 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs text-slate-400">
              <div className="flex items-center gap-3">
                <ShieldCheck className="h-5 w-5 text-emerald-400 shrink-0" />
                <span>
                  <strong>Zero-Retention Privacy:</strong> Scans are processed 100% in-memory and immediately discarded. No patient data or images are ever stored on disk.
                </span>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* Global Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-600">
        <div className="max-w-7xl mx-auto px-4 space-y-1">
          <p>© 2026 AI-Based Brain Tumor Classification, Segmentation and Explainable MRI Analysis</p>
          <p className="text-[11px] text-slate-700">Research prototype based on BRISC2025 benchmark. Strictly for scientific exploration.</p>
        </div>
      </footer>

    </div>
  );
}
