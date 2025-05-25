import pytest
import torch
from pathlib import Path
import json
import yaml
from core.training.train_lora import LoRATrainer, TextDataset
from core.workflows.workflow_manager import WorkflowManager, create_basic_workflow
from core.utils.common import (
    setup_directories,
    load_config,
    save_config,
    get_device,
    set_seed
)

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "output_dir": "models/trained_loras",
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "lora_rank": 4,
        "lora_alpha": 32.0
    }

@pytest.fixture
def sample_data():
    """Create sample training data."""
    return [
        {
            "text": "A test image description",
            "metadata": {"style": "test"}
        },
        {
            "text": "Another test image description",
            "metadata": {"style": "test"}
        }
    ]

def test_setup_directories(temp_dir):
    """Test directory setup."""
    dirs = setup_directories(str(temp_dir))
    assert all(d.exists() for d in dirs.values())
    assert all(d.is_dir() for d in dirs.values())

def test_config_operations(temp_dir, sample_config):
    """Test configuration loading and saving."""
    config_path = temp_dir / "test_config.yaml"
    
    # Test saving config
    save_config(sample_config, str(config_path))
    assert config_path.exists()
    
    # Test loading config
    loaded_config = load_config(str(config_path))
    assert loaded_config == sample_config

def test_device_setup():
    """Test device setup."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu"]

def test_seed_setting():
    """Test random seed setting."""
    set_seed(42)
    # Generate some random numbers
    rand1 = torch.rand(1)
    set_seed(42)
    rand2 = torch.rand(1)
    assert torch.allclose(rand1, rand2)

def test_workflow_manager(temp_dir):
    """Test workflow management functionality."""
    manager = WorkflowManager(str(temp_dir / "workflows"))
    
    # Test workflow creation
    workflow_data = create_basic_workflow()
    workflow_id = manager.create_workflow(
        name=workflow_data["name"],
        description=workflow_data["description"],
        nodes=workflow_data["nodes"],
        edges=workflow_data["edges"]
    )
    
    # Test workflow retrieval
    retrieved_workflow = manager.get_workflow(workflow_id)
    assert retrieved_workflow["name"] == workflow_data["name"]
    
    # Test workflow listing
    workflows = manager.list_workflows()
    assert len(workflows) == 1
    assert workflows[0]["id"] == workflow_id
    
    # Test workflow update
    update_data = {"description": "Updated description"}
    updated_workflow = manager.update_workflow(workflow_id, update_data)
    assert updated_workflow["description"] == "Updated description"
    
    # Test workflow deletion
    manager.delete_workflow(workflow_id)
    workflows = manager.list_workflows()
    assert len(workflows) == 0

def test_text_dataset(temp_dir, sample_data):
    """Test text dataset functionality."""
    # Save sample data
    data_path = temp_dir / "test_data.json"
    with open(data_path, 'w') as f:
        json.dump(sample_data, f)
    
    # Create dataset
    tokenizer = torch.nn.Module()  # Mock tokenizer
    dataset = TextDataset(str(data_path), tokenizer)
    
    assert len(dataset) == len(sample_data)
    # Note: We can't test __getitem__ without a real tokenizer

def test_lora_trainer(sample_config):
    """Test LoRA trainer initialization."""
    # This is a basic test that only checks initialization
    # Full training tests would require more setup and time
    trainer = LoRATrainer(sample_config)
    assert trainer.config == sample_config
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'tokenizer') 