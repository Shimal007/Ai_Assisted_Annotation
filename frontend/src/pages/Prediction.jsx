import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, Download, Settings, Eye, AlertCircle, Loader } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const YOLODetectionTool = () => {
  const [selectedImages, setSelectedImages] = useState([]);
  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.3);
  const [modelStatus, setModelStatus] = useState({ loaded: false, type: null });
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [classes, setClasses] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/model/status`);
        if (!response.ok) throw new Error('Failed to fetch model status');
        const data = await response.json();
        setModelStatus({
          loaded: data.models_loaded,
          type: data.available_models?.join(', ') || 'N/A',
        });
      } catch (err) {
        setError('Failed to connect to backend: ' + err.message);
      }
    };

    const fetchClasses = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/classes`);
        if (!response.ok) throw new Error('Failed to fetch classes');
        const data = await response.json();
        setClasses(data.classes || []);
      } catch (err) {
        setError('Failed to fetch classes: ' + err.message);
      }
    };

    checkModelStatus();
    fetchClasses();
  }, []);

  const runDetection = async (imageFiles) => {
    setIsLoading(true);
    setError(null);
    setDetections([]);

    try {
      const formData = new FormData();
      formData.append('image', imageFiles[0].file);
      formData.append('confidence', confidenceThreshold.toString());
      formData.append('model_choice', selectedModel);

      const response = await fetch(`${API_BASE_URL}/api/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(`Detection failed: ${errData.error || response.statusText}`);
      }

      const result = await response.json();
      
      if (result.error) {
          throw new Error(result.error);
      }
      
      const processedDetections = result.detections.map((det, index) => ({
        id: `${result.image_info.filename}-${index + 1}`,
        class: det.class,
        confidence: det.confidence,
        bbox: {
          x: det.bbox[0] - det.bbox[2] / 2,
          y: det.bbox[1] - det.bbox[3] / 2,
          width: det.bbox[2],
          height: det.bbox[3],
          angle: det.bbox[4] || 0,
        },
        filename: result.image_info.filename,
      }));
      setDetections(processedDetections);

    } catch (err) {
      setError(err.message);
      console.error('Detection error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const drawDetections = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !selectedImages.length) return;

    const ctx = canvas.getContext('2d');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);

    const currentImage = selectedImages[0];
    const imageDetections = detections.filter((det) => det.filename === currentImage.file.name);

    imageDetections.forEach((detection) => {
        const { bbox, class: className, confidence } = detection;

        // --- 1. Draw the rotated bounding box ---
        ctx.save();
        ctx.translate(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        ctx.rotate((bbox.angle * Math.PI) / 180);
        ctx.strokeStyle = '#FF0000'; // <-- CHANGED: Set box color to red
        ctx.lineWidth = Math.max(4, image.naturalWidth / 400);
        ctx.strokeRect(-bbox.width / 2, -bbox.height / 2, bbox.width, bbox.height);
        ctx.restore();

        // --- 2. Draw the label (class + confidence) ---
        const label = `${className} ${(confidence * 100).toFixed(0)}%`;
        
        const fontSize = Math.max(16, image.naturalWidth / 80);
        ctx.font = `bold ${fontSize}px Arial`;
        
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = fontSize * 1.2;

        const labelX = bbox.x;
        const labelY = bbox.y - textHeight - ctx.lineWidth;

        ctx.fillStyle = '#FF0000B3'; // <-- CHANGED: Set label background to semi-transparent red
        ctx.fillRect(labelX, labelY, textWidth + 10, textHeight);

        ctx.fillStyle = 'white';
        ctx.fillText(label, labelX + 5, labelY + fontSize);
    });
  };
  
  useEffect(() => {
    const image = imageRef.current;
    if (image && selectedImages.length > 0) {
      const handleLoad = () => drawDetections();
      image.addEventListener('load', handleLoad);
      if (image.complete) handleLoad();
      return () => image.removeEventListener('load', handleLoad);
    }
  }, [detections, selectedImages]);

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    if (files.length) {
      const images = files.map((file) => ({
        file,
        url: URL.createObjectURL(file),
      }));
      setSelectedImages(images);
      setDetections([]);
      setError(null);
    }
  };

  const handleDetection = () => {
    if (selectedImages.length && modelStatus.loaded) {
      runDetection(selectedImages);
    }
  };

  const handleDownload = () => {
    if (!canvasRef.current || !selectedImages.length) return;
    const link = document.createElement('a');
    link.download = `annotated_${selectedImages[0].file.name}`;
    link.href = canvasRef.current.toDataURL('image/jpeg', 0.9);
    link.click();
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl mb-6 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl"><Eye className="w-8 h-8 text-white" /></div>
              <div>
                <h1 className="text-3xl font-bold text-gray-800">YOLOv8-OBB Detection Tool</h1>
                <p className="text-gray-600">AI-Assisted Aerial Image Annotation</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button onClick={() => setShowSettings(!showSettings)} className="p-3 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">
                <Settings className="w-6 h-6 text-gray-600" />
              </button>
              <div className={`px-4 py-2 rounded-full text-sm font-medium ${modelStatus.loaded ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                {modelStatus.loaded ? '✓ Model Ready' : '⚠ Model Not Loaded'}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Upload Image(s)</h3>
              <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleImageUpload} className="hidden" />
              <button onClick={() => fileInputRef.current?.click()} className="w-full border-2 border-dashed border-gray-300 hover:border-blue-500 rounded-xl p-8 transition-colors duration-200 flex flex-col items-center space-y-3">
                <Upload className="w-8 h-8 text-gray-400" />
                <span className="text-gray-600">Click to upload</span>
              </button>
              {selectedImages.length > 0 && (
                <div className="mt-4 text-sm text-green-800 font-medium p-3 bg-green-50 rounded-lg">✓ {selectedImages.length} image(s) selected.</div>
              )}
            </div>
            
            {showSettings && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Detection Settings</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Confidence Threshold: {confidenceThreshold.toFixed(2)}</label>
                    <input type="range" min="0.1" max="0.95" step="0.05" value={confidenceThreshold} onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                  </div>
                  
                  <div>
                    <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-2">Detection Model</label>
                    <select
                      id="model-select"
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="ensemble">Ensemble (Balanced)</option>
                      <option value="yolov8x-obb.pt">YOLOv8x (Max Accuracy)</option>
                      <option value="yolov8l-obb.pt">YOLOv8l (High Accuracy)</option>
                      <option value="yolov8m-obb.pt">YOLOv8m (Fastest)</option>
                    </select>
                  </div>

                  {classes.length > 0 && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Supported Classes ({classes.length})</label>
                      <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto bg-gray-50 p-2 rounded">
                        {classes.map((cls, index) => (<div key={index} className="text-xs text-gray-600 capitalize truncate">{cls}</div>))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {selectedImages.length > 0 && (
              <button onClick={handleDetection} disabled={isLoading || !modelStatus.loaded} className="w-full bg-gradient-to-r from-green-500 to-blue-500 text-white py-4 px-6 rounded-xl hover:from-green-600 hover:to-blue-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 text-lg font-medium">
                {isLoading ? <Loader className="w-6 h-6 animate-spin" /> : <Camera className="w-6 h-6" />}
                <span>{isLoading ? 'Detecting...' : 'Run Detection'}</span>
              </button>
            )}
            
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">{error}</div>
            )}
            
            {detections.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Results ({detections.length})</h3>
                  <button onClick={handleDownload} disabled={!selectedImages[0]} className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50">
                    <Download className="w-4 h-4" />
                    <span>Export</span>
                  </button>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {detections.map((detection) => (
                    <div key={detection.id} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg border">
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#FF0000' }}></div>
                        <span className="font-medium text-gray-800 capitalize text-sm">{detection.class}</span>
                      </div>
                      <span className={`font-bold text-sm ${getConfidenceColor(detection.confidence)}`}>{(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              {!selectedImages.length ? (
                <div className="flex flex-col items-center justify-center h-96 text-gray-500">
                  <Camera className="w-24 h-24 mb-4 text-gray-300" />
                  <h3 className="text-xl font-medium">No Image Selected</h3>
                  <p>Upload an image to start detection.</p>
                </div>
              ) : (
                <div className="relative">
                  <img ref={imageRef} src={selectedImages[0].url} alt="Uploaded" className="absolute inset-0 w-full h-full object-contain opacity-0 pointer-events-none" />
                  <canvas ref={canvasRef} className="w-full h-auto max-h-[80vh] object-contain" />
                  {isLoading && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                      <div className="bg-white rounded-xl p-6 flex items-center space-x-4">
                        <Loader className="w-8 h-8 animate-spin text-blue-500" />
                        <span className="text-gray-800 font-medium">Processing...</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default YOLODetectionTool;