from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, List
import os
import asyncio
import aiofiles
import hashlib
from datetime import datetime
import logging
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil
import warnings
import requests
from PIL import Image
import numpy as np

# Hugging Face imports
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Advanced Deepfake Detection API",
    description="Professional deepfake and AI-generated image detection using Hugging Face Deep-Fake-Detector-v2",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "temp_uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

class DeepfakeRequest(BaseModel):
    image_url: Optional[HttpUrl] = None
    context: Optional[str] = ""

class DetectionResponse(BaseModel):
    is_fake: bool
    confidence: float
    prediction_scores: Dict[str, float]
    predicted_class: str
    verdict: str
    risk_level: str
    processing_time: float
    timestamp: str
    metadata: Optional[Dict] = None
    model_info: Dict[str, str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_name: str
    device: str

class HuggingFaceDeepfakeDetector:
    """
    Professional deepfake detector using Hugging Face pre-trained model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        
        logger.info(f"Loading Hugging Face model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Initialize the model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the Hugging Face model and processor"""
        try:
            # Method 1: Using pipeline (simpler)
            self.pipeline = pipeline(
                "image-classification", 
                model=self.model_name,
                device=0 if self.device.type == 'cuda' else -1
            )
            
            # Method 2: Direct model loading (for more control)
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Hugging Face model loaded successfully!")
            
            # Test the model with a dummy prediction to warm it up
            self._warmup_model()
            
        except Exception as e:
            logger.error(f"❌ Failed to load Hugging Face model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _warmup_model(self):
        """Warm up the model with a dummy prediction"""
        try:
            # Create a dummy image
            dummy_image = Image.new('RGB', (224, 224), color='white')
            
            # Test pipeline
            result = self.pipeline(dummy_image)
            logger.info(f"🔥 Model warmed up successfully! Test result: {result[0]['label']}")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def detect_with_pipeline(self, image_path: str) -> Dict:
        """Detect using the pipeline method (recommended)"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Run prediction
            results = self.pipeline(image)
            
            # Parse results - the model returns list of predictions
            # Format: [{'label': 'REAL'/'FAKE', 'score': confidence}, ...]
            predictions = {item['label']: item['score'] for item in results}
            
            # Determine the predicted class and confidence
            predicted_class = max(predictions, key=predictions.get)
            max_confidence = predictions[predicted_class]
            
            # Determine if fake
            is_fake = predicted_class.upper() in ['FAKE', 'ARTIFICIAL', 'GENERATED', 'DEEPFAKE']
            
            return {
                'method': 'pipeline',
                'is_fake': is_fake,
                'predicted_class': predicted_class,
                'confidence': max_confidence,
                'all_predictions': predictions,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Pipeline detection error: {e}")
            return {
                'method': 'pipeline',
                'success': False,
                'error': str(e),
                'is_fake': None,
                'confidence': 0.0
            }
    
    def detect_with_model(self, image_path: str) -> Dict:
        """Detect using direct model inference"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get class labels
            predicted_class_id = predictions.argmax().item()
            confidence = predictions[0][predicted_class_id].item()
            
            # Map to labels (this depends on how the model was trained)
            # Common mappings: 0=REAL, 1=FAKE or 0=FAKE, 1=REAL
            id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else {0: 'REAL', 1: 'FAKE'}
            predicted_class = id2label[predicted_class_id]
            
            # Create prediction scores dict
            prediction_scores = {}
            for class_id, class_name in id2label.items():
                prediction_scores[class_name] = predictions[0][class_id].item()
            
            is_fake = predicted_class.upper() in ['FAKE', 'ARTIFICIAL', 'GENERATED', 'DEEPFAKE']
            
            return {
                'method': 'direct_model',
                'is_fake': is_fake,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': prediction_scores,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Direct model detection error: {e}")
            return {
                'method': 'direct_model',
                'success': False,
                'error': str(e),
                'is_fake': None,
                'confidence': 0.0
            }
    
    async def detect_deepfake(self, image_path: str) -> Dict:
        """Main detection method with both approaches"""
        import time
        start_time = time.time()
        
        logger.info(f"🔍 Analyzing image: {image_path}")
        
        def _detect():
            # Try pipeline first (more reliable)
            pipeline_result = self.detect_with_pipeline(image_path)
            
            if pipeline_result['success']:
                return pipeline_result
            else:
                # Fallback to direct model
                logger.warning("Pipeline failed, trying direct model...")
                return self.detect_with_model(image_path)
        
        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        detection_result = await loop.run_in_executor(executor, _detect)
        
        processing_time = time.time() - start_time
        
        if not detection_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Detection failed: {detection_result.get('error', 'Unknown error')}"
            )
        
        # Calculate risk level
        confidence = detection_result['confidence']
        risk_level = self._calculate_risk_level(detection_result['is_fake'], confidence)
        
        # Prepare final result
        result = {
            'is_fake': detection_result['is_fake'],
            'confidence': float(confidence),
            'prediction_scores': detection_result['all_predictions'],
            'predicted_class': detection_result['predicted_class'],
            'verdict': 'FAKE' if detection_result['is_fake'] else 'REAL',
            'risk_level': risk_level,
            'processing_time': float(processing_time),
            'timestamp': datetime.utcnow().isoformat(),
            'model_info': {
                'name': self.model_name,
                'method': detection_result['method'],
                'version': '2.0.0'
            }
        }
        
        logger.info(f"✅ Detection completed: {result['verdict']} (confidence: {confidence:.3f})")
        return result
    
    def _calculate_risk_level(self, is_fake: bool, confidence: float) -> str:
        """Calculate risk level based on prediction and confidence"""
        if is_fake:
            if confidence > 0.9:
                return 'VERY_HIGH'
            elif confidence > 0.75:
                return 'HIGH'
            elif confidence > 0.6:
                return 'MEDIUM'
            else:
                return 'LOW'
        else:  # Real image
            if confidence > 0.9:
                return 'VERY_LOW'
            elif confidence > 0.75:
                return 'LOW'
            elif confidence > 0.6:
                return 'MEDIUM'
            else:
                return 'HIGH'  # Low confidence in "real" = suspicious

# Initialize detector
detector = HuggingFaceDeepfakeDetector()

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.filename:
        return False
    
    # Check extension
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        return False
    
    return True

async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file temporarily"""
    file_hash = hashlib.md5(f"{upload_file.filename}{datetime.now()}".encode()).hexdigest()
    ext = os.path.splitext(upload_file.filename.lower())[1]
    temp_filename = f"{file_hash}{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)
    
    return temp_path

async def download_image_from_url(url: str) -> str:
    """Download image from URL"""
    try:
        response = requests.get(str(url), timeout=15, stream=True)
        response.raise_for_status()
        
        # Generate temp filename
        file_hash = hashlib.md5(str(url).encode()).hexdigest()
        temp_filename = f"{file_hash}.jpg"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to download image from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def cleanup_temp_file(file_path: str):
    """Background task to cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🧹 Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0.0",
        model_name=detector.model_name,
        device=str(detector.device)
    )

@app.post("/detect/upload", response_model=DetectionResponse)
async def detect_deepfake_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    context: str = Form(default="")
):
    """
    Detect deepfake from uploaded image file using Hugging Face model
    """
    # Validate file
    if not validate_image_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file. Supported formats: JPG, PNG, WebP, BMP. Max size: 10MB"
        )
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = await save_upload_file(file)
        
        # Perform detection
        result = await detector.detect_deepfake(temp_path)
        
        # Add metadata
        result['metadata'] = {
            'filename': file.filename,
            'file_size': file.size,
            'content_type': file.content_type,
            'context': context
        }
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return DetectionResponse(**result)
        
    except Exception as e:
        # Cleanup on error
        if temp_path:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/url", response_model=DetectionResponse)
async def detect_deepfake_url(
    background_tasks: BackgroundTasks,
    request: DeepfakeRequest
):
    """
    Detect deepfake from image URL using Hugging Face model
    """
    if not request.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required")
    
    temp_path = None
    try:
        # Download image
        temp_path = await download_image_from_url(request.image_url)
        
        # Perform detection
        result = await detector.detect_deepfake(temp_path)
        
        # Add metadata
        result['metadata'] = {
            'source_url': str(request.image_url),
            'context': request.context or ""
        }
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return DetectionResponse(**result)
        
    except Exception as e:
        # Cleanup on error
        if temp_path:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        logger.error(f"URL detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch")
async def detect_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Batch detection for multiple images using Hugging Face model
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    temp_paths = []
    
    try:
        # Process all files
        for file in files:
            if not validate_image_file(file):
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file format or size',
                    'status': 'failed'
                })
                continue
            
            try:
                # Save and process file
                temp_path = await save_upload_file(file)
                temp_paths.append(temp_path)
                
                result = await detector.detect_deepfake(temp_path)
                result['filename'] = file.filename
                result['status'] = 'success'
                results.append(result)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Schedule cleanup for all temp files
        for temp_path in temp_paths:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return {
            'total_files': len(files),
            'processed': len([r for r in results if r.get('status') == 'success']),
            'failed': len([r for r in results if r.get('status') == 'failed']),
            'results': results,
            'timestamp': datetime.utcnow().isoformat(),
            'model_info': {
                'name': detector.model_name,
                'version': '2.0.0'
            }
        }
        
    except Exception as e:
        # Cleanup on error
        for temp_path in temp_paths:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    return {
        'model_name': detector.model_name,
        'model_type': 'Hugging Face Transformers',
        'task': 'image-classification',
        'description': 'Professional deepfake detection using pre-trained transformer model',
        'version': '2.0.0',
        'device': str(detector.device),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
        'capabilities': [
            'Image upload detection',
            'URL-based detection',
            'Batch processing',
            'Real-time inference',
            'High accuracy classification'
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics and information"""
    return {
        'api_info': {
            'version': '2.0.0',
            'model': detector.model_name,
            'device': str(detector.device)
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'limits': {
            'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
            'max_batch_size': 10
        },
        'endpoints': [
            'POST /detect/upload - Upload image detection',
            'POST /detect/url - URL image detection', 
            'POST /detect/batch - Batch image detection',
            'GET /health - Health check',
            'GET /model/info - Model information',
            'GET /stats - API statistics'
        ],
        'timestamp': datetime.utcnow().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Startup event - model is already loaded in constructor"""
    logger.info("🚀 Advanced Deepfake Detection API Started!")
    logger.info(f"📱 Model: {detector.model_name}")
    logger.info(f"💻 Device: {detector.device}")
    logger.info("✅ Ready to detect deepfakes!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down...")
    # Cleanup temp directory
    try:
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        logger.info("🧹 Temporary files cleaned up")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "fake:app",  # Replace "main" with your filename
        host="0.0.0.0",
        port=8001,
        reload=True,
        workers=1  # Use 1 worker for GPU models
    )