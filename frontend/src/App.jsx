import React, { useState, useEffect, useRef } from 'react';
import { 
  UploadCloud, 
  RefreshCw, 
  Check, 
  AlertCircle, 
  ArrowRight,
  Info,
  Shield,
  Layers,
  Eye,
  Activity,
  X,
  FileText
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const DEMO_SAMPLES = [
  { id: 'glioma', label: 'Glioma', path: '/samples/glioma_sample.jpg', hint: 'Frontal infiltration' },
  { id: 'meningioma', label: 'Meningioma', path: '/samples/meningioma_sample.jpg', hint: 'Dural extra-axial' },
  { id: 'pituitary', label: 'Pituitary', path: '/samples/pituitary_sample.jpg', hint: 'Sellar lesion' },
  { id: 'no_tumor', label: 'No Tumor', path: '/samples/no_tumor_sample.jpg', hint: 'Normal anatomy' },
];

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('all');
  const [selectedDemo, setSelectedDemo] = useState(null);
  const fileInputRef = useRef(null);
  const uploadSectionRef = useRef(null);
  const resultsSectionRef = useRef(null);

  const loadingSteps = [
    'Decoding MRI slice in memory (256×256)...',
    'Extracting ResNet-18 feature representations...',
    'Generating layer4 Grad-CAM attribution map...',
    'Predicting U-Net boundary delineation (Epoch 23)...',
    'Compiling analysis report and confidence distribution...'
  ];

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(selected.type)) {
      setError('Please upload an image in JPG, JPEG, or PNG format.');
      return;
    }
    if (selected.size > 10 * 1024 * 1024) {
      setError('The image exceeds the 10 MB file size limit.');
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
      if (!res.ok) throw new Error('Failed to load sample image');
      const blob = await res.blob();
      const demoFile = new File([blob], `${demo.id}_sample.jpg`, { type: 'image/jpeg' });
      setFile(demoFile);
      setPreviewUrl(demo.path);
    } catch {
      setError('Failed to load the selected sample scan.');
    }
  };

  const scrollToUpload = () => {
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleAnalyze = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError(null);
    setLoadingStep(0);

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
        throw new Error(errData.detail || `Inference error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      setActiveTab('all');

      setTimeout(() => {
        resultsSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);

    } catch (err) {
      clearInterval(stepInterval);
      setError(err.message || 'An error occurred during analysis.');
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
    scrollToUpload();
  };

  return (
    <div className="min-h-screen bg-[#151313] text-[#F2EEEE] flex flex-col font-sans selection:bg-[#C94A50]/25 selection:text-white">
      
      {/* ============================================================ */}
      {/* 1. NAVIGATION                                                */}
      {/* ============================================================ */}
      <header className="border-b border-[#2C2626] bg-[#151313]/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          
          {/* Logo */}
          <div className="flex items-center gap-3">
            <a href="#" className="flex items-center gap-2 text-white font-semibold text-base tracking-tight hover:text-white">
              <span className="w-2.5 h-2.5 rounded-full bg-[#C94A50]"></span>
              <span>NeuroVision</span>
            </a>
            <span className="hidden sm:inline-block text-[11px] font-medium text-[#B7ADAD] border-l border-[#393030] pl-3 ml-1">
              Brain MRI Analysis
            </span>
          </div>

          {/* Links */}
          <nav className="flex items-center gap-6 text-sm text-[#B7ADAD]">
            <a href="#hero" className="hover:text-[#F2EEEE] transition-colors">Overview</a>
            <a href="#how-it-works" className="hover:text-[#F2EEEE] transition-colors">How it works</a>
            <button 
              onClick={scrollToUpload}
              className="hover:text-[#F2EEEE] transition-colors"
            >
              Analyze MRI
            </button>
            {result && (
              <a href="#results-section" className="text-[#C94A50] font-medium transition-colors">
                Results
              </a>
            )}
          </nav>

        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-10 space-y-16">

        {/* ============================================================ */}
        {/* 2. HERO SECTION                                              */}
        {/* ============================================================ */}
        <section id="hero" className="pt-4 sm:pt-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
            
            {/* Left Content */}
            <div className="lg:col-span-7 space-y-5">
              
              <div className="inline-block">
                <span className="text-xs font-medium text-[#B7ADAD] tracking-wider uppercase bg-[#1D1A1A] border border-[#2F2727] px-2.5 py-1 rounded">
                  AI-Assisted Medical Imaging Research
                </span>
              </div>

              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-semibold text-[#F2EEEE] tracking-tight leading-tight">
                Brain MRI Analysis
              </h1>

              <p className="text-base sm:text-lg text-[#B7ADAD] leading-relaxed max-w-xl">
                AI-assisted classification, tumor-region segmentation, and visual explanation of brain MRI scans.
              </p>

              <div className="pt-2 flex flex-wrap items-center gap-4">
                <button
                  id="btn-hero-cta"
                  onClick={scrollToUpload}
                  className="btn-primary px-5 py-2.5 text-sm flex items-center gap-2"
                >
                  <span>Analyze an MRI</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
                <a
                  href="#how-it-works"
                  className="btn-secondary px-4 py-2.5 text-sm"
                >
                  How it works
                </a>
              </div>

              {/* Research Framework Specs */}
              <div className="pt-6 border-t border-[#2C2626] grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                <div>
                  <span className="text-[#857A7A] block">Target Classes</span>
                  <span className="text-[#F2EEEE] font-medium">Glioma, Men., Pit., None</span>
                </div>
                <div>
                  <span className="text-[#857A7A] block">Classifier</span>
                  <span className="text-[#F2EEEE] font-medium">ResNet-18</span>
                </div>
                <div>
                  <span className="text-[#857A7A] block">Explainability</span>
                  <span className="text-[#F2EEEE] font-medium">Grad-CAM (layer4)</span>
                </div>
                <div>
                  <span className="text-[#857A7A] block">Segmentation</span>
                  <span className="text-[#F2EEEE] font-medium">4-Level U-Net</span>
                </div>
              </div>

            </div>

            {/* Right: Authentic Brain MRI Presentation */}
            <div className="lg:col-span-5">
              <div className="card-surface p-4 max-w-sm mx-auto shadow-sm">
                <div className="aspect-square bg-black rounded overflow-hidden border border-[#2F2727]">
                  <img 
                    src="/samples/glioma_sample.jpg" 
                    alt="Sample Brain MRI Scan" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="mt-3 flex items-center justify-between text-xs text-[#B7ADAD]">
                  <span>Sample T1 axial MRI scan</span>
                  <span className="text-[#857A7A]">256 × 256</span>
                </div>
              </div>
            </div>

          </div>
        </section>

        {/* ============================================================ */}
        {/* 3. HOW IT WORKS                                             */}
        {/* ============================================================ */}
        <section id="how-it-works" className="pt-6 border-t border-[#2C2626] space-y-6">
          
          <div>
            <h2 className="text-xl sm:text-2xl font-semibold text-[#F2EEEE] tracking-tight">
              How the analysis works
            </h2>
            <p className="text-sm text-[#B7ADAD] mt-1">
              A structured four-stage computational pipeline for research investigation.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            
            <div className="card-surface p-5 space-y-2">
              <div className="text-xs font-semibold text-[#C94A50]">01</div>
              <h3 className="text-sm font-semibold text-[#F2EEEE]">MRI input</h3>
              <p className="text-xs text-[#B7ADAD] leading-relaxed">
                Upload a T1-weighted axial brain MRI scan (JPG, PNG). The slice is normalized strictly in memory without permanent disk storage.
              </p>
            </div>

            <div className="card-surface p-5 space-y-2">
              <div className="text-xs font-semibold text-[#C94A50]">02</div>
              <h3 className="text-sm font-semibold text-[#F2EEEE]">Tumor classification</h3>
              <p className="text-xs text-[#B7ADAD] leading-relaxed">
                A ResNet-18 classifier predicts category probabilities across four classes: Glioma, Meningioma, Pituitary, or No Tumor.
              </p>
            </div>

            <div className="card-surface p-5 space-y-2">
              <div className="text-xs font-semibold text-[#C94A50]">03</div>
              <h3 className="text-sm font-semibold text-[#F2EEEE]">Grad-CAM visualization</h3>
              <p className="text-xs text-[#B7ADAD] leading-relaxed">
                Backpropagated gradient activations from layer4 generate a coarse heatmap highlighting anatomical regions influencing the prediction.
              </p>
            </div>

            <div className="card-surface p-5 space-y-2">
              <div className="text-xs font-semibold text-[#C94A50]">04</div>
              <h3 className="text-sm font-semibold text-[#F2EEEE]">U-Net segmentation</h3>
              <p className="text-xs text-[#B7ADAD] leading-relaxed">
                A 4-level U-Net model predicts pixel-level tumor boundaries on positive cases, computing the lesion area percentage at threshold ≥ 0.50.
              </p>
            </div>

          </div>

        </section>

        {/* ============================================================ */}
        {/* 4. MRI UPLOAD INTERFACE                                      */}
        {/* ============================================================ */}
        <section id="upload-section" ref={uploadSectionRef} className="pt-6 border-t border-[#2C2626] space-y-6">
          
          <div>
            <h2 className="text-xl sm:text-2xl font-semibold text-[#F2EEEE] tracking-tight">
              Upload MRI for analysis
            </h2>
            <p className="text-sm text-[#B7ADAD] mt-1">
              Select an image from your device or test with a validated benchmark scan.
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-4 rounded-md bg-[#231A1A] border border-[#87383D] flex items-start gap-3 text-sm text-[#F2D0D2]">
              <AlertCircle className="w-5 h-5 text-[#C94A50] shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-medium text-[#F2EEEE]">Analysis Request Failed</p>
                <p className="text-xs text-[#B7ADAD] mt-0.5">{error}</p>
              </div>
              <button onClick={() => setError(null)} className="text-[#857A7A] hover:text-[#F2EEEE]">
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          <div className="card-surface p-6 sm:p-8 space-y-6 max-w-3xl mx-auto">
            
            {/* Dropzone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`border border-dashed rounded-md p-8 text-center cursor-pointer transition-colors ${
                previewUrl 
                  ? 'border-[#4F4343] bg-[#1D1A1A]' 
                  : 'border-[#393030] hover:border-[#4F4343] bg-[#1D1A1A]/60 hover:bg-[#1D1A1A]'
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
                  <div className="relative inline-block bg-black rounded overflow-hidden border border-[#393030]">
                    <img 
                      src={previewUrl} 
                      alt="Selected MRI Scan" 
                      className="h-56 w-56 object-cover object-center"
                    />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[#F2EEEE]">{file?.name}</p>
                    <p className="text-xs text-[#B7ADAD] mt-0.5">Click to select a different scan</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-3 py-4">
                  <div className="w-10 h-10 mx-auto rounded-full bg-[#272222] border border-[#393030] flex items-center justify-center text-[#B7ADAD]">
                    <UploadCloud className="w-5 h-5" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-[#F2EEEE]">
                      Upload a brain MRI image
                    </p>
                    <p className="text-xs text-[#B7ADAD]">
                      Supported formats: JPG, JPEG, PNG • Maximum size: 10 MB
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Quick Benchmark Samples */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs text-[#B7ADAD]">
                <span className="font-medium">Or select a preloaded sample scan:</span>
                <span className="text-[#857A7A]">Standard benchmark</span>
              </div>
              
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {DEMO_SAMPLES.map((demo) => {
                  const isSelected = selectedDemo === demo.id;
                  return (
                    <button
                      key={demo.id}
                      id={`demo-${demo.id}`}
                      onClick={() => handleSelectDemo(demo)}
                      className={`p-3 text-left rounded-md border text-xs transition-colors ${
                        isSelected
                          ? 'bg-[#291F20] border-[#C94A50] text-white'
                          : 'bg-[#1D1A1A] border-[#393030] text-[#B7ADAD] hover:bg-[#252020] hover:text-[#F2EEEE]'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-[#F2EEEE]">{demo.label}</span>
                        {isSelected && <Check className="w-3.5 h-3.5 text-[#C94A50]" />}
                      </div>
                      <span className="text-[11px] text-[#857A7A] block mt-0.5 line-clamp-1">{demo.hint}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Action Button */}
            <div className="pt-2">
              <button
                id="btn-analyze-mri"
                disabled={!file || loading}
                onClick={handleAnalyze}
                className="btn-primary w-full py-3 text-sm flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Processing in-memory pipeline...</span>
                  </>
                ) : (
                  <>
                    <span>Analyze MRI</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>

            {/* Loading Feedback */}
            {loading && (
              <div className="card-surface-subtle p-4 rounded-md space-y-3">
                <div className="flex items-center justify-between text-xs text-[#B7ADAD]">
                  <span>Step {loadingStep + 1} of {loadingSteps.length}</span>
                  <span className="text-[#857A7A]">Executing</span>
                </div>
                <div className="w-full bg-[#151313] h-1.5 rounded-full overflow-hidden border border-[#2F2727]">
                  <div 
                    className="bg-[#C94A50] h-full transition-all duration-300"
                    style={{ width: `${((loadingStep + 1) / loadingSteps.length) * 100}%` }}
                  ></div>
                </div>
                <p className="text-xs text-[#F2EEEE]">
                  {loadingSteps[loadingStep]}
                </p>
              </div>
            )}

          </div>

        </section>

        {/* ============================================================ */}
        {/* 5. ANALYSIS RESULTS                                          */}
        {/* ============================================================ */}
        {result && (
          <section id="results-section" ref={resultsSectionRef} className="pt-6 border-t border-[#2C2626] space-y-8">
            
            <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#2C2626]">
              <div>
                <h2 className="text-xl sm:text-2xl font-semibold text-[#F2EEEE] tracking-tight">
                  Analysis results
                </h2>
                <p className="text-xs text-[#B7ADAD] mt-0.5">
                  Synthesized outputs from ResNet-18 classification, layer4 Grad-CAM, and U-Net segmentation.
                </p>
              </div>

              <button
                id="btn-reset-scan"
                onClick={handleReset}
                className="btn-secondary px-3.5 py-2 text-xs flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Analyze another scan</span>
              </button>
            </div>

            {/* Metrics Overview */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              
              {/* Card 1: Primary AI Prediction */}
              <div className="lg:col-span-5 card-surface p-6 space-y-4">
                
                <div className="flex items-center justify-between text-xs text-[#B7ADAD]">
                  <span>AI-predicted category</span>
                  <span className="text-[11px] bg-[#1D1A1A] border border-[#2F2727] px-2 py-0.5 rounded text-[#B7ADAD]">
                    ResNet-18
                  </span>
                </div>

                <div>
                  <div className="text-2xl sm:text-3xl font-semibold text-[#F2EEEE] tracking-tight">
                    {result.prediction.class}
                  </div>
                  <div className="mt-1 flex items-baseline gap-2 text-sm">
                    <span className="font-semibold text-[#C94A50]">
                      {result.prediction.confidence_percentage}%
                    </span>
                    <span className="text-xs text-[#857A7A]">model confidence</span>
                  </div>
                </div>

                <p className="text-xs text-[#857A7A] border-t border-[#2C2626] pt-3 leading-relaxed">
                  AI prediction for research prototype evaluation. Not a clinically validated diagnosis.
                </p>

                {/* Region status */}
                <div className="pt-2 flex items-center justify-between text-xs border-t border-[#2C2626]">
                  <span className="text-[#B7ADAD]">Predicted Region:</span>
                  <span className="font-medium text-[#F2EEEE]">
                    {result.prediction.raw_class === 'no_tumor' || result.segmentation.is_non_lesional
                      ? 'Non-lesional (0.0% area)'
                      : `${result.segmentation.tumor_area_percentage}% of slice area`}
                  </span>
                </div>

              </div>

              {/* Card 2: Probability Distribution */}
              <div className="lg:col-span-7 card-surface p-6 space-y-4">
                
                <div className="flex items-center justify-between text-xs text-[#B7ADAD]">
                  <span className="font-medium">Category probability distribution</span>
                  <span className="text-[#857A7A]">Softmax output</span>
                </div>

                <div className="space-y-3 pt-1">
                  {Object.entries(result.probabilities).map(([clsName, prob]) => {
                    const pct = (prob * 100).toFixed(1);
                    const isTop = clsName === result.prediction.class;

                    return (
                      <div key={clsName} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className={isTop ? 'font-medium text-white' : 'text-[#B7ADAD]'}>
                            {clsName}
                          </span>
                          <span className={isTop ? 'font-medium text-[#C94A50]' : 'text-[#857A7A]'}>
                            {pct}%
                          </span>
                        </div>
                        <div className="w-full bg-[#151313] h-2 rounded-full overflow-hidden border border-[#2F2727]">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              isTop ? 'bg-[#C94A50]' : 'bg-[#403737]'
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

            {/* Non-Lesional Safety Notice (For No Tumor) */}
            {(result.prediction.raw_class === 'no_tumor' || result.segmentation.is_non_lesional) && (
              <div className="card-surface-subtle p-4 rounded-md flex items-start gap-3 text-xs text-[#B7ADAD] border border-[#393030]">
                <Info className="w-4 h-4 text-[#B7ADAD] shrink-0 mt-0.5" />
                <div className="space-y-0.5">
                  <p className="font-medium text-[#F2EEEE]">Non-Lesional Scan Protocol</p>
                  <p className="leading-relaxed">
                    The classification network categorized this scan as <strong>No Tumor</strong>. The U-Net segmentation model was trained specifically on tumor-positive scans and does not delineate non-lesional cases (assigned region area: 0.0%). The absence of a predicted region is not proof that a tumor is absent.
                  </p>
                </div>
              </div>
            )}

            {/* Visualizations Panel */}
            <div className="card-surface rounded-md overflow-hidden space-y-6">
              
              {/* Tabs */}
              <div className="px-6 pt-4 border-b border-[#2C2626] flex flex-wrap gap-2">
                {[
                  { id: 'all', label: 'Multi-modal overview', icon: Layers },
                  { id: 'gradcam', label: 'Grad-CAM', icon: Eye },
                  { id: 'unet', label: 'U-Net segmentation', icon: Activity },
                ].map((tab) => {
                  const IconComp = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      id={`tab-${tab.id}`}
                      onClick={() => setActiveTab(tab.id)}
                      className={`pb-3 px-3 text-xs font-medium border-b-2 flex items-center gap-2 transition-colors ${
                        isActive
                          ? 'border-[#C94A50] text-white'
                          : 'border-transparent text-[#B7ADAD] hover:text-[#F2EEEE]'
                      }`}
                    >
                      <IconComp className="w-3.5 h-3.5" />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* Tab 1: All 4 Panels */}
              {activeTab === 'all' && (
                <div className="p-6 pt-0 space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    
                    {/* Panel 1 */}
                    <div className="bg-[#1D1A1A] p-3 rounded border border-[#2F2727] space-y-2">
                      <div className="flex items-center justify-between text-xs text-[#F2EEEE]">
                        <span className="font-medium">Original MRI</span>
                        <span className="text-[11px] text-[#857A7A]">256×256</span>
                      </div>
                      <div className="aspect-square bg-black rounded overflow-hidden border border-[#252020]">
                        <img 
                          src={result.visualizations.original_mri} 
                          alt="Original MRI" 
                          className="w-full h-full object-cover" 
                        />
                      </div>
                      <p className="text-[11px] text-[#857A7A]">Preprocessed axial slice</p>
                    </div>

                    {/* Panel 2 */}
                    <div className="bg-[#1D1A1A] p-3 rounded border border-[#2F2727] space-y-2">
                      <div className="flex items-center justify-between text-xs text-[#F2EEEE]">
                        <span className="font-medium">Grad-CAM heatmap</span>
                        <span className="text-[11px] text-[#857A7A]">layer4</span>
                      </div>
                      <div className="aspect-square bg-black rounded overflow-hidden border border-[#252020]">
                        <img 
                          src={result.visualizations.gradcam_heatmap} 
                          alt="Grad-CAM Heatmap" 
                          className="w-full h-full object-cover" 
                        />
                      </div>
                      <p className="text-[11px] text-[#857A7A]">Feature gradient saliency</p>
                    </div>

                    {/* Panel 3 */}
                    <div className="bg-[#1D1A1A] p-3 rounded border border-[#2F2727] space-y-2">
                      <div className="flex items-center justify-between text-xs text-[#F2EEEE]">
                        <span className="font-medium">Grad-CAM overlay</span>
                        <span className="text-[11px] text-[#857A7A]">Attention</span>
                      </div>
                      <div className="aspect-square bg-black rounded overflow-hidden border border-[#252020]">
                        <img 
                          src={result.visualizations.gradcam_overlay} 
                          alt="Grad-CAM Overlay" 
                          className="w-full h-full object-cover" 
                        />
                      </div>
                      <p className="text-[11px] text-[#857A7A]">Spatial visual attribution</p>
                    </div>

                    {/* Panel 4 */}
                    <div className="bg-[#1D1A1A] p-3 rounded border border-[#2F2727] space-y-2">
                      <div className="flex items-center justify-between text-xs text-[#F2EEEE]">
                        <span className="font-medium">Predicted segmentation</span>
                        <span className="text-[11px] text-[#857A7A]">Epoch 23</span>
                      </div>
                      <div className="aspect-square bg-black rounded overflow-hidden border border-[#252020]">
                        <img 
                          src={result.visualizations.segmentation_overlay} 
                          alt="U-Net Segmentation Overlay" 
                          className="w-full h-full object-cover" 
                        />
                      </div>
                      <p className="text-[11px] text-[#857A7A]">
                        {result.prediction.raw_class === 'no_tumor' || result.segmentation.is_non_lesional
                          ? 'Non-lesional: 0.0% area'
                          : `Predicted region: ${result.segmentation.tumor_area_percentage}%`}
                      </p>
                    </div>

                  </div>
                </div>
              )}

              {/* Tab 2: Grad-CAM Detail */}
              {activeTab === 'gradcam' && (
                <div className="p-6 pt-0 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                    
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-[#B7ADAD] block">
                        Grad-CAM visual attention overlay
                      </span>
                      <div className="aspect-square bg-black rounded overflow-hidden border border-[#393030]">
                        <img 
                          src={result.visualizations.gradcam_overlay} 
                          alt="Grad-CAM Overlay" 
                          className="w-full h-full object-cover" 
                        />
                      </div>
                    </div>

                    <div className="space-y-4 text-xs sm:text-sm text-[#B7ADAD] leading-relaxed bg-[#1D1A1A] p-6 rounded border border-[#2F2727]">
                      <h4 className="text-base font-semibold text-[#F2EEEE]">
                        Grad-CAM explainability
                      </h4>
                      <p>
                        Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradients of the predicted category score (<strong>{result.prediction.class}</strong>) with respect to the final convolutional feature maps in <strong>ResNet-18 (layer4)</strong>.
                      </p>
                      <p>
                        Warm regions indicate anatomical textures and spatial areas that positively influenced the classifier's category assignment.
                      </p>
                      <div className="p-3 bg-[#151313] rounded border border-[#2C2626] text-xs space-y-1">
                        <p className="font-medium text-[#F2EEEE]">Scientific Interpretation Notice:</p>
                        <p className="text-[#857A7A]">Grad-CAM represents coarse visual feature attribution and does not prove causal reasoning, clinical validity, or surgical margins.</p>
                      </div>
                    </div>

                  </div>
                </div>
              )}

              {/* Tab 3: U-Net Detail */}
              {activeTab === 'unet' && (
                <div className="p-6 pt-0 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                    
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-[#B7ADAD] block">
                        U-Net boundary delineation overlay
                      </span>
                      <div className="aspect-square bg-black rounded overflow-hidden border border-[#393030]">
                        <img 
                          src={result.visualizations.segmentation_overlay} 
                          alt="U-Net Overlay" 
                          className="w-full h-full object-cover" 
                        />
                      </div>
                    </div>

                    <div className="space-y-4 text-xs sm:text-sm text-[#B7ADAD] leading-relaxed bg-[#1D1A1A] p-6 rounded border border-[#2F2727]">
                      <h4 className="text-base font-semibold text-[#F2EEEE]">
                        4-Level U-Net segmentation
                      </h4>
                      <p>
                        The 4-level U-Net model predicts pixel-level binary tumor boundaries from single-channel 256×256 normalized MRI slices using an activation threshold of <strong>{result.segmentation.threshold_used}</strong>.
                      </p>
                      
                      {result.prediction.raw_class === 'no_tumor' || result.segmentation.is_non_lesional ? (
                        <div className="p-3 bg-[#151313] rounded border border-[#2C2626] text-xs text-[#B7ADAD] space-y-1">
                          <p className="font-medium text-[#F2EEEE]">Non-Lesional Result</p>
                          <p className="text-[#857A7A]">The classifier identified this scan as non-lesional. U-Net was trained specifically on tumor cases; therefore, non-lesional scans have no assigned tumor boundaries.</p>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 gap-3 pt-2">
                          <div className="p-3 bg-[#151313] rounded border border-[#2C2626]">
                            <span className="text-[11px] text-[#857A7A] block">Predicted region area</span>
                            <span className="text-base font-semibold text-[#F2EEEE]">{result.segmentation.tumor_area_percentage}%</span>
                          </div>
                          <div className="p-3 bg-[#151313] rounded border border-[#2C2626]">
                            <span className="text-[11px] text-[#857A7A] block">Decision threshold</span>
                            <span className="text-base font-semibold text-[#F2EEEE]">&ge; 0.50</span>
                          </div>
                        </div>
                      )}

                      <div className="p-3 bg-[#151313] rounded border border-[#2C2626] text-xs text-[#857A7A]">
                        <strong>No Fabricated Metrics:</strong> Ground-truth evaluation metrics (such as Dice or IoU) are not computed or displayed for unlabelled user-uploaded scans.
                      </div>
                    </div>

                  </div>
                </div>
              )}

            </div>

          </section>
        )}

        {/* ============================================================ */}
        {/* 6. MEDICAL DISCLAIMER & PRIVACY                              */}
        {/* ============================================================ */}
        <section className="border-t border-[#2C2626] pt-10 pb-6 space-y-4">
          
          <div className="card-surface-subtle p-6 rounded-md space-y-3">
            
            <div className="flex items-center gap-2 text-[#C94A50]">
              <Shield className="w-4 h-4" />
              <h3 className="font-medium text-xs uppercase tracking-wider text-[#F2EEEE]">
                Research Prototype Disclaimer & Privacy Notice
              </h3>
            </div>

            <p className="text-xs text-[#B7ADAD] leading-relaxed">
              “This application is an AI-assisted research prototype and is not a clinically validated diagnostic system. Its predictions and visualizations should not be used for diagnosis, treatment decisions, or surgical planning. Please consult a qualified medical professional for medical interpretation.”
            </p>

            <div className="pt-3 border-t border-[#2A2424] grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs text-[#857A7A]">
              <div>
                <strong className="text-[#B7ADAD] block font-medium">In-Memory Processing</strong>
                <span>All uploaded scans are processed strictly in memory and discarded immediately upon response synthesis. No scans or patient data are permanently stored or logged.</span>
              </div>
              <div>
                <strong className="text-[#B7ADAD] block font-medium">Independent Execution</strong>
                <span>Inference is executed locally or via dedicated containers. No images are transmitted to external third-party AI APIs or analytics tracking services.</span>
              </div>
            </div>

          </div>

        </section>

      </main>

      {/* ============================================================ */}
      {/* 7. FOOTER                                                    */}
      {/* ============================================================ */}
      <footer className="border-t border-[#2C2626] bg-[#121010] py-6 text-center text-xs text-[#857A7A]">
        <div className="max-w-6xl mx-auto px-4 space-y-1">
          <p className="text-[#B7ADAD]">
            NeuroVision • AI-Based Brain Tumor Classification, Segmentation and Explainable MRI Analysis
          </p>
          <p className="text-[11px]">
            ResNet-18 Backbone • Layer4 Grad-CAM • 4-Level U-Net • Evaluated on BRISC2025 Benchmark
          </p>
        </div>
      </footer>

    </div>
  );
}
