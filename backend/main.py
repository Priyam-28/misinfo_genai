from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import re
import logging
from typing import List, Dict, Any, Optional
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
tokenizer = None
model = None
explainer = None
device = None

class TextToAnalyze(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze for credibility")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()

class ExplanationItem(BaseModel):
    token: str = Field(..., description="Token/word from the text")
    weight: float = Field(..., description="Influence weight of the token")

class AnalysisResponse(BaseModel):
    credibility_score: float = Field(..., description="Probability that the text is real (0-1)")
    confidence: float = Field(..., description="Model confidence in the prediction (0-1)")
    label: str = Field(..., description="Classification label: 'likely_real' or 'likely_fake'")
    explanation: List[ExplanationItem] = Field(..., description="Token-level explanations")
    processing_time: float = Field(..., description="Time taken for analysis in seconds")
    model_info: Dict[str, str] = Field(..., description="Information about the model used")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    message: str

async def load_model_components():
    """Load model, tokenizer, and explainer components"""
    global tokenizer, model, explainer, device
    
    try:
        logger.info("Loading model components...")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        model_name = "hamzab/roberta-fake-news-classification"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        # Initialize LIME explainer
        explainer = LimeTextExplainer(
            class_names=["Fake", "Real"],
            feature_selection='auto'
        )
        
        logger.info("Model components loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model_components()
    yield
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="API for detecting fake news using transformer models with explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', ' ', text)
    return text.strip()

def predictor(texts: List[str]) -> np.ndarray:
    """
    Prediction function for LIME
    
    Args:
        texts: List of text strings to classify
        
    Returns:
        numpy array of prediction probabilities
    """
    try:
        # Tokenize inputs
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.cpu().detach().numpy()
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise e

def get_model_confidence(probs: np.ndarray) -> float:
    """Calculate model confidence based on prediction probabilities"""
    # Confidence is the absolute difference from 0.5 (neutral)
    max_prob = np.max(probs)
    confidence = abs(max_prob - 0.5) * 2  # Scale to 0-1
    return float(confidence)

@app.get("/", response_model=Dict[str, str])
async def read_root():
    """Root endpoint"""
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = all([tokenizer is not None, model is not None, explainer is not None])
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device=str(device) if device else "unknown",
        message="All systems operational" if model_loaded else "Model not loaded"
    )

@app.post("/analyze_text", response_model=AnalysisResponse)
async def analyze_text(text_data: TextToAnalyze):
    """
    Analyze text for fake news detection
    
    Args:
        text_data: TextToAnalyze object containing the text to analyze
        
    Returns:
        AnalysisResponse with credibility score, label, and explanations
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received text analysis request: {len(text_data.text) if text_data.text else 0} characters")
        # Check if model is loaded
        if not all([tokenizer, model, explainer]):
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Clean the input text
        text = clean_text(text_data.text)
        
        if len(text) < 10:
            raise HTTPException(status_code=400, detail="Text too short for meaningful analysis")
        
        logger.info(f"Analyzing text of length: {len(text)}")
        
        # Get prediction
        prediction_probs = predictor([text])[0]
        prediction = np.argmax(prediction_probs)
        
        # Calculate scores
        credibility_score = float(prediction_probs[1])  # Probability of being real
        confidence = get_model_confidence(prediction_probs)
        label = "likely_real" if prediction == 1 else "likely_fake"
        
        # Generate explanation using LIME
        try:
            explanation = explainer.explain_instance(
                text, 
                predictor, 
                num_features=min(10, len(text.split())),  # Limit features based on text length
                num_samples=100  # Reduce for faster processing
            )
            explanation_list = explanation.as_list()
            
            # Format explanation for response
            formatted_explanation = [
                ExplanationItem(token=str(token), weight=float(weight)) 
                for token, weight in explanation_list
            ]
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {str(e)}")
            # Fallback: return empty explanation
            formatted_explanation = []
        
        processing_time = time.time() - start_time
        
        response = AnalysisResponse(
            credibility_score=credibility_score,
            confidence=confidence,
            label=label,
            explanation=formatted_explanation,
            processing_time=processing_time,
            model_info={
                "model_name": "hamzab/roberta-fake-news-classification",
                "framework": "transformers",
                "device": str(device)
            }
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s - Label: {label}, Score: {credibility_score:.3f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

@app.post("/batch_analyze")
async def batch_analyze_text(texts: List[str]):
    """
    Analyze multiple texts at once (without LIME explanations for efficiency)
    
    Args:
        texts: List of text strings to analyze
        
    Returns:
        List of basic analysis results
    """
    try:
        if not all([tokenizer, model]):
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if len(texts) > 50:
            raise HTTPException(status_code=400, detail="Too many texts (max 50)")
        
        # Clean texts
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Get predictions for all texts
        prediction_probs = predictor(cleaned_texts)
        
        results = []
        for i, (text, probs) in enumerate(zip(cleaned_texts, prediction_probs)):
            prediction = np.argmax(probs)
            credibility_score = float(probs[1])
            confidence = get_model_confidence(probs)
            label = "likely_real" if prediction == 1 else "likely_fake"
            
            results.append({
                "index": i,
                "credibility_score": credibility_score,
                "confidence": confidence,
                "label": label,
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            })
        
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during batch analysis")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)