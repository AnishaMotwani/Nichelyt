
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json

from backend.schemas import AnalyzeRequest
from nichelyt_pipeline import run_nichelyt_pipeline


app = FastAPI(
    title="Nichelyt API",
    description="Backend API for Nichelyt Instagram Content Analysis",
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to Nichelyt API",
        "status": "running"
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Nichelyt backend is working"
    }


# Main analysis endpoint
@app.post("/analyze")
def analyze_content(request: AnalyzeRequest):

    try:

        result = run_nichelyt_pipeline(

            niche=request.niche,
            goal=request.goal,
            post_idea=request.post_idea,

            followers=request.followers,
            post_type=request.post_type,
            caption=request.caption,

            post_date=request.post_date,
            post_time=request.post_time,

            run_counterfactual=request.run_counterfactual,
            run_shap=request.run_shap,
            run_gemini=request.run_gemini,

            tone=request.tone,
            target_audience=request.target_audience,
            language=request.language

        )

        # Check whether pipeline returned a dictionary
        if not isinstance(result, dict):

            raise HTTPException(
                status_code=500,
                detail="Pipeline returned an invalid response."
            )


        # Handle pipeline errors
        if result.get("status") == "error":

            raise HTTPException(
                status_code=400,
                detail=result
            )


        # Convert non-JSON-compatible values
        # such as NumPy values into JSON-compatible values
        json_safe_result = json.loads(
            json.dumps(
                result,
                default=str
            )
        )


        return json_safe_result


    except HTTPException:
        raise


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(error)
            }
        )