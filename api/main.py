from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="PromptWeaver API",
    description="API for managing AI models and ComfyUI workflows",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkflowConfig(BaseModel):
    name: str
    description: Optional[str] = None
    nodes: List[dict]
    edges: List[dict]

class TrainingConfig(BaseModel):
    model_name: str
    dataset_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_rank: int = 4
    lora_alpha: float = 32.0

@app.get("/")
async def root():
    return {"message": "Welcome to PromptWeaver API"}

@app.post("/workflows/")
async def create_workflow(config: WorkflowConfig):
    try:
        # TODO: Implement workflow creation logic
        return {"message": "Workflow created successfully", "workflow_id": "123"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/lora")
async def train_lora(config: TrainingConfig):
    try:
        # TODO: Implement LoRA training logic
        return {"message": "Training started", "training_id": "456"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # TODO: Implement dataset upload logic
        return {"message": "Dataset uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/")
async def list_models():
    try:
        # TODO: Implement model listing logic
        return {"models": ["model1", "model2"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 