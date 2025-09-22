from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
import os
import asyncio
import uvicorn
from io import BytesIO
import base64

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient

warnings.filterwarnings('ignore')

# FastAPI app initialization
app = FastAPI(
    title="AI Fact-Checker API with Web Search",
    description="Comprehensive fact-checking service using LangChain + Gemini + Web Search",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ClaimRequest(BaseModel):
    claim: str = Field(..., description="The claim to fact-check", min_length=5)
    use_search: bool = Field(default=True, description="Whether to use web search for verification")

class BatchRequest(BaseModel):
    text: str = Field(..., description="Text containing multiple claims to analyze", min_length=10)
    show_progress: bool = Field(default=False, description="Whether to show progress updates")
    use_search: bool = Field(default=True, description="Whether to use web search for verification")

class FactCheckResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    explanation: str
    key_facts: Optional[List[str]] = []
    sources: Optional[List[str]] = []
    context: Optional[str] = ""
    search_results: Optional[List[Dict[str, Any]]] = []
    timestamp: str
    used_search: bool

class BatchResponse(BaseModel):
    total_claims: int
    results: List[FactCheckResponse]
    summary: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    search_available: bool

# Enhanced LangChain Fact-Checker Class with Web Search
class LangChainFactChecker:
    def __init__(self, gemini_api_key: str, tavily_api_key: str = None):
        """Initialize the fact-checker with LangChain + Google GenAI + Web Search"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp", 
                google_api_key=gemini_api_key, 
                temperature=0.2
            )
            
            # Initialize web search
            self.search_available = False
            if tavily_api_key:
                try:
                    self.tavily_client = TavilyClient(api_key="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")
                    self.search_tool = TavilySearchResults(
                        api_wrapper=self.tavily_client,
                        max_results=5,
                        search_depth="advanced"
                    )
                    self.search_available = True
                    print("✅ Tavily Search initialized successfully!")
                except Exception as e:
                    print(f"⚠ Tavily Search failed to initialize: {e}")
                    self.search_available = False
            
            self.initialized = True
            print("✅ Gemini Flash (via LangChain) initialized successfully!")
        except Exception as e:
            self.initialized = False
            print(f"❌ Failed to initialize Gemini: {e}")
            raise e

    async def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information about a claim"""
        if not self.search_available:
            return []
        
        try:
            # Use Tavily for web search
            search_results = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False
            )
            
            # Format results
            formatted_results = []
            for result in search_results.get('results', []):
                formatted_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', '')[:500] + '...',  # Limit content length
                    'score': result.get('score', 0.0)
                })
            
            return formatted_results
        except Exception as e:
            print(f"⚠ Web search error: {e}")
            return []

    async def extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text"""
        prompt = ChatPromptTemplate.from_template("""
        Extract individual factual claims from the following text.
        Return them strictly as a JSON list of strings.
        Focus on verifiable factual statements, ignore opinions.
        
        Text: "{text}"
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            claims = await asyncio.to_thread(chain.invoke, {"text": text})
            return claims if isinstance(claims, list) else [text]
        except Exception as e:
            print(f"⚠ Error extracting claims: {e}")
            # Fallback: split by sentences
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10][:10]  # Limit to 10
    
    async def fact_check_claim(self, claim: str, use_search: bool = True) -> Dict[str, Any]:
        """Comprehensive fact-check of a single claim with optional web search"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Search for relevant information if requested and available
        search_results = []
        search_context = ""
        
        if use_search and self.search_available:
            print(f"🔍 Searching web for: {claim[:50]}...")
            search_results = await self.search_web(claim)
            
            if search_results:
                # Create search context for the LLM
                search_context = "\n\nRELEVANT WEB SEARCH RESULTS:\n"
                for i, result in enumerate(search_results[:3], 1):  # Use top 3 results
                    search_context += f"{i}. {result['title']} ({result['url']})\n"
                    search_context += f"   Content: {result['content']}\n\n"
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert fact-checker with access to the latest knowledge. 
        Today is {date}.
        
        VERIFY THIS CLAIM: "{claim}"
        
        {search_context}
        
        Based on your knowledge and the search results above (if provided), provide analysis in this EXACT JSON format:
        {{
            "claim": "{claim}",
            "verdict": "TRUE/FALSE/PARTIALLY_TRUE/MISLEADING/UNVERIFIABLE",
            "confidence": 0.95,
            "explanation": "detailed analysis with evidence and reasoning, referencing search results when available",
            "key_facts": ["fact 1", "fact 2", "fact 3"],
            "sources": ["credible source 1", "credible source 2"],
            "context": "important context, caveats, or nuances"
        }}
        
        Be thorough but concise. Use confidence scores from 0.0 to 1.0.
        If search results are provided, reference them in your explanation.
        If search results contradict your knowledge, note this in the explanation.
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = await asyncio.to_thread(chain.invoke, {
                "claim": claim, 
                "date": current_date,
                "search_context": search_context
            })
            
            # Ensure required fields exist
            if not isinstance(result, dict):
                raise ValueError("Invalid response format")
                
            result.setdefault('key_facts', [])
            result.setdefault('sources', [])
            result.setdefault('context', '')
            result['timestamp'] = datetime.now().isoformat()
            result['search_results'] = search_results
            result['used_search'] = use_search and bool(search_results)
            
            return result
        except Exception as e:
            return {
                "claim": claim,
                "verdict": "ERROR",
                "confidence": 0.0,
                "explanation": f"Processing error: {str(e)}",
                "key_facts": [],
                "sources": [],
                "context": "Error occurred during analysis",
                "search_results": search_results,
                "used_search": use_search and bool(search_results),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_batch(self, text: str, show_progress: bool = False, use_search: bool = True) -> Dict[str, Any]:
        """Analyze multiple claims and return structured results"""
        print("🔍 Starting batch analysis...")
        
        # Extract claims
        claims = await self.extract_claims(text)
        total_claims = len(claims)
        print(f"📋 Found {total_claims} claims to verify")
        
        if total_claims > 20:  # Limit for API safety
            claims = claims[:20]
            print(f"⚠ Limited to first 20 claims for API performance")
        
        results = []
        for i, claim in enumerate(claims, 1):
            if show_progress:
                print(f"⚡ Analyzing {i}/{len(claims)}: {claim[:60]}...")
            
            result = await self.fact_check_claim(claim, use_search)
            results.append(result)
            
            # Rate limiting - longer delay when using search
            delay = 2 if use_search else 1
            if i < len(claims):
                await asyncio.sleep(delay)
        
        # Create summary statistics
        df = pd.DataFrame(results)
        verdict_counts = df['verdict'].value_counts().to_dict()
        avg_confidence = float(df['confidence'].mean()) if len(df) > 0 else 0.0
        search_usage = sum(1 for r in results if r.get('used_search', False))
        
        summary = {
            "total_processed": len(results),
            "verdict_distribution": verdict_counts,
            "average_confidence": round(avg_confidence, 3),
            "search_usage": f"{search_usage}/{len(results)} claims used web search",
            "processing_time": f"{len(claims)} claims processed"
        }
        
        print(f"✅ Batch analysis complete!")
        return {
            "total_claims": len(results),
            "results": results,
            "summary": summary
        }

# Global fact-checker instance
fact_checker = None

# Startup event
@app.on_event("startup")
async def startup_event():
    global fact_checker
    
    # Get API keys from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY",)
    tavily_api_key = os.getenv("TAVILY_API_KEY",)  # Set this environment variable
    
    if not gemini_api_key or gemini_api_key.startswith("YOUR_"):
        raise ValueError("❌ No valid Gemini API key found. Set GEMINI_API_KEY environment variable.")
    
    if not tavily_api_key:
        print("⚠ No Tavily API key found. Web search will be disabled.")
        print("💡 Set TAVILY_API_KEY environment variable to enable web search.")
        print("💡 Get your free API key at: https://tavily.com")
    
    try:
        fact_checker = LangChainFactChecker(gemini_api_key, tavily_api_key)
        print("🚀 Enhanced Fact-Checker API ready!")
        if fact_checker.search_available:
            print("🌐 Web search enabled!")
        else:
            print("📚 Running in knowledge-only mode")
    except Exception as e:
        print(f"❌ Failed to initialize fact-checker: {e}")
        raise e

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Fact-Checker API with Web Search",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
        "search_enabled": str(fact_checker.search_available if fact_checker else False)
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if fact_checker and fact_checker.initialized else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        search_available=fact_checker.search_available if fact_checker else False
    )

@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_single_claim(request: ClaimRequest):
    """Fact-check a single claim with optional web search"""
    if not fact_checker or not fact_checker.initialized:
        raise HTTPException(status_code=503, detail="Fact-checker service unavailable")
    
    try:
        result = await fact_checker.fact_check_claim(request.claim, request.use_search)
        return FactCheckResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing claim: {str(e)}")

@app.post("/fact-check/batch", response_model=BatchResponse)
async def fact_check_batch(request: BatchRequest):
    """Fact-check multiple claims from text with optional web search"""
    if not fact_checker or not fact_checker.initialized:
        raise HTTPException(status_code=503, detail="Fact-checker service unavailable")
    
    try:
        result = await fact_checker.analyze_batch(request.text, request.show_progress, request.use_search)
        
        # Convert results to response models
        fact_check_results = [FactCheckResponse(**res) for res in result["results"]]
        
        return BatchResponse(
            total_claims=result["total_claims"],
            results=fact_check_results,
            summary=result["summary"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/search/{query}")
async def search_web_direct(query: str, max_results: int = 5):
    """Direct web search endpoint for testing"""
    if not fact_checker or not fact_checker.search_available:
        raise HTTPException(status_code=503, detail="Web search service unavailable")
    
    try:
        results = await fact_checker.search_web(query, max_results)
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get general statistics about the service"""
    return {
        "service": "AI Fact-Checker with Web Search",
        "model": "Gemini-2.0-Flash-Exp via LangChain",
        "search_engine": "Tavily",
        "supported_verdicts": ["TRUE", "FALSE", "PARTIALLY_TRUE", "MISLEADING", "UNVERIFIABLE"],
        "max_batch_size": 20,
        "rate_limit": "2 seconds per claim with search, 1 second without",
        "search_available": fact_checker.search_available if fact_checker else False
    }

@app.post("/analyze/claims")
async def analyze_text_for_claims(request: BatchRequest):
    """Extract claims from text without fact-checking"""
    if not fact_checker or not fact_checker.initialized:
        raise HTTPException(status_code=503, detail="Fact-checker service unavailable")
    
    try:
        claims = await fact_checker.extract_claims(request.text)
        return {
            "original_text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "extracted_claims": claims,
            "total_claims": len(claims),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting claims: {str(e)}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {"error": "Invalid input", "detail": str(exc)}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}

# Main function to run the server
if __name__ == "__main__":
    uvicorn.run(
        "verify:app",  # Change "main" to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )