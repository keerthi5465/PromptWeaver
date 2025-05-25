import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
import yaml
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowManager:
    def __init__(self, workflows_dir: str = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
    def create_workflow(self, name: str, nodes: List[Dict], edges: List[Dict], description: Optional[str] = None) -> str:
        """Create a new ComfyUI workflow."""
        workflow_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow_data = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "nodes": nodes,
            "edges": edges,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        workflow_path = self.workflows_dir / f"{workflow_id}.json"
        with open(workflow_path, 'w') as f:
            json.dump(workflow_data, f, indent=2)
            
        logger.info(f"Created workflow: {workflow_id}")
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> Dict:
        """Retrieve a workflow by ID."""
        workflow_path = self.workflows_dir / f"{workflow_id}.json"
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {workflow_id} not found")
            
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    def list_workflows(self) -> List[Dict]:
        """List all available workflows."""
        workflows = []
        for workflow_file in self.workflows_dir.glob("*.json"):
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
                workflows.append({
                    "id": workflow_data["id"],
                    "name": workflow_data["name"],
                    "description": workflow_data["description"],
                    "created_at": workflow_data["created_at"]
                })
        return workflows
    
    def update_workflow(self, workflow_id: str, updates: Dict) -> Dict:
        """Update an existing workflow."""
        workflow_path = self.workflows_dir / f"{workflow_id}.json"
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {workflow_id} not found")
            
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        # Update workflow data
        workflow_data.update(updates)
        workflow_data["updated_at"] = datetime.now().isoformat()
        
        with open(workflow_path, 'w') as f:
            json.dump(workflow_data, f, indent=2)
            
        logger.info(f"Updated workflow: {workflow_id}")
        return workflow_data
    
    def delete_workflow(self, workflow_id: str) -> None:
        """Delete a workflow."""
        workflow_path = self.workflows_dir / f"{workflow_id}.json"
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {workflow_id} not found")
            
        workflow_path.unlink()
        logger.info(f"Deleted workflow: {workflow_id}")
    
    def export_workflow(self, workflow_id: str, format: str = "json") -> str:
        """Export a workflow in the specified format."""
        workflow_data = self.get_workflow(workflow_id)
        
        if format == "json":
            return json.dumps(workflow_data, indent=2)
        elif format == "yaml":
            return yaml.dump(workflow_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_workflow(self, workflow_data: Dict) -> str:
        """Import a workflow from a dictionary."""
        if "id" not in workflow_data:
            workflow_id = f"{workflow_data['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            workflow_data["id"] = workflow_id
        
        workflow_path = self.workflows_dir / f"{workflow_data['id']}.json"
        with open(workflow_path, 'w') as f:
            json.dump(workflow_data, f, indent=2)
            
        logger.info(f"Imported workflow: {workflow_data['id']}")
        return workflow_data['id']

def create_basic_workflow() -> Dict:
    """Create a basic ComfyUI workflow template."""
    return {
        "name": "Basic Image Generation",
        "description": "A basic workflow for image generation using Stable Diffusion",
        "nodes": [
            {
                "id": "1",
                "type": "CLIPTextEncode",
                "inputs": {
                    "text": "A beautiful landscape",
                    "clip": ["2", 0]
                }
            },
            {
                "id": "2",
                "type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                }
            },
            {
                "id": "3",
                "type": "KSampler",
                "inputs": {
                    "seed": 1234,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["2", 0],
                    "positive": ["1", 0],
                    "negative": ["1", 0],
                    "latent_image": ["4", 0]
                }
            },
            {
                "id": "4",
                "type": "EmptyLatentImage",
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                }
            }
        ],
        "edges": [
            {"from": "2", "to": "1", "fromOutput": 0, "toInput": 1},
            {"from": "2", "to": "3", "fromOutput": 0, "toInput": 5},
            {"from": "1", "to": "3", "fromOutput": 0, "toInput": 6},
            {"from": "1", "to": "3", "fromOutput": 0, "toInput": 7},
            {"from": "4", "to": "3", "fromOutput": 0, "toInput": 8}
        ]
    }

if __name__ == "__main__":
    # Example usage
    manager = WorkflowManager()
    
    # Create a basic workflow
    workflow_data = create_basic_workflow()
    workflow_id = manager.create_workflow(
        name=workflow_data["name"],
        description=workflow_data["description"],
        nodes=workflow_data["nodes"],
        edges=workflow_data["edges"]
    )
    
    # List all workflows
    workflows = manager.list_workflows()
    print(f"Available workflows: {workflows}")
    
    # Export the workflow
    exported = manager.export_workflow(workflow_id, format="yaml")
    print(f"Exported workflow:\n{exported}") 