"""FastAPI application for Virtual Try-On AI."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import config
from backend.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup and shutdown."""
    logger.info("Starting Virtual Try-On AI application")
    # Startup
    yield
    # Shutdown
    logger.info("Shutting down Virtual Try-On AI application")


# Create FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>Virtual Try-On AI</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .endpoint { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
                code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🎨 Virtual Try-On AI</h1>
            <p>Welcome to the Virtual Try-On AI API!</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <strong>POST /api/virtual-tryon</strong><br>
                Generate a virtual try-on result<br>
                <code>Expected: person_image_base64, garment_image_base64, garment_type (optional)</code>
            </div>
            
            <div class="endpoint">
                <strong>POST /api/virtual-tryon-upload</strong><br>
                Generate virtual try-on with file uploads<br>
                <code>Expected: person_image (file), garment_image (file), garment_type (optional)</code>
            </div>
            
            <div class="endpoint">
                <strong>POST /api/classify-garment</strong><br>
                Classify clothing type from an image<br>
                <code>Expected: image_base64</code>
            </div>
            
            <div class="endpoint">
                <strong>GET /api/health</strong><br>
                Health check endpoint
            </div>
            
            <div class="endpoint">
                <strong>POST /api/clear-cache</strong><br>
                Clear GPU/CPU cache
            </div>
            
            <h2>Documentation:</h2>
            <ul>
                <li><a href="/docs">Swagger UI</a> - Interactive API documentation</li>
                <li><a href="/redoc">ReDoc</a> - ReDoc documentation</li>
            </ul>
            
            <hr>
            <p><small>Virtual Try-On AI - AI-powered virtual clothing try-on</small></p>
        </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Virtual Try-On AI API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.API_LOG_LEVEL,
    )
