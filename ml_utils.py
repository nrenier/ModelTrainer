import os
import logging
import zipfile
import tarfile
import json
import yaml
import shutil
from app import app

logger = logging.getLogger(__name__)

def process_dataset(file_path, dataset_format):
    """
    Process an uploaded dataset file.
    Extract if needed, validate the structure, and return dataset information.
    
    Args:
        file_path: Path to the uploaded file
        dataset_format: Format of the dataset (COCO, YOLO, etc.)
        
    Returns:
        dict: Information about the dataset (num_classes, num_images, etc.)
    """
    # Create a directory for extraction if needed
    dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(os.path.basename(file_path))[0])
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Extract the file if it's an archive
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(dataset_dir)
    elif file_path.endswith('.tar'):
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(dataset_dir)
    else:
        # If it's not an archive, just copy the file to the dataset directory
        shutil.copy2(file_path, dataset_dir)
    
    # Now parse the dataset according to its format
    if dataset_format.upper() == 'COCO':
        return parse_coco_dataset(dataset_dir)
    elif dataset_format.upper() == 'YOLO':
        return parse_yolo_dataset(dataset_dir)
    elif dataset_format.upper() == 'PASCAL VOC':
        return parse_pascal_voc_dataset(dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

def parse_coco_dataset(dataset_dir):
    """
    Parse a COCO format dataset and extract information.
    
    Args:
        dataset_dir: Directory containing the extracted dataset
        
    Returns:
        dict: Information about the dataset
    """
    # Look for annotations file
    annotation_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    
    if not annotation_files:
        # Check if there's an annotations subdirectory
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        if os.path.isdir(annotations_dir):
            annotation_files = [os.path.join('annotations', f) for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    if not annotation_files:
        raise ValueError("No COCO annotation file found in the dataset")
    
    # Use the first annotation file found
    annotation_file = os.path.join(dataset_dir, annotation_files[0])
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Extract dataset information
    categories = coco_data.get('categories', [])
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    
    class_names = [cat['name'] for cat in categories]
    
    return {
        'num_classes': len(categories),
        'num_images': len(images),
        'num_annotations': len(annotations),
        'class_names': class_names
    }

def parse_yolo_dataset(dataset_dir):
    """
    Parse a YOLO format dataset and extract information.
    
    Args:
        dataset_dir: Directory containing the extracted dataset
        
    Returns:
        dict: Information about the dataset
    """
    # Look for data.yaml file
    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    if not os.path.exists(data_yaml):
        # Look for other YAML files that might contain dataset info
        yaml_files = [f for f in os.listdir(dataset_dir) if f.endswith('.yaml') or f.endswith('.yml')]
        if yaml_files:
            data_yaml = os.path.join(dataset_dir, yaml_files[0])
        else:
            raise ValueError("No data.yaml or equivalent file found in YOLO dataset")
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract class names
    class_names = data.get('names', [])
    if isinstance(class_names, dict):
        # Convert from dict format {0: 'person', 1: 'car', ...} to list
        max_id = max(class_names.keys())
        class_list = [''] * (max_id + 1)
        for idx, name in class_names.items():
            class_list[idx] = name
        class_names = class_list
    
    # Count images in train, val and test directories
    num_images = 0
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split, 'images')
        if os.path.isdir(split_dir):
            num_images += len([f for f in os.listdir(split_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    return {
        'num_classes': len(class_names),
        'num_images': num_images,
        'class_names': class_names
    }

def parse_pascal_voc_dataset(dataset_dir):
    """
    Parse a Pascal VOC format dataset and extract information.
    
    Args:
        dataset_dir: Directory containing the extracted dataset
        
    Returns:
        dict: Information about the dataset
    """
    # Look for annotations directory
    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    if not os.path.isdir(annotations_dir):
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        if not os.path.isdir(annotations_dir):
            raise ValueError("No Annotations directory found in Pascal VOC dataset")
    
    # Count annotation files (XML files)
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    # Extract class names by parsing XML files
    import xml.etree.ElementTree as ET
    class_names = set()
    
    # Parse a few files to get class names
    for xml_file in annotation_files[:min(100, len(annotation_files))]:
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        for obj in root.findall('.//object'):
            name = obj.find('name').text
            class_names.add(name)
    
    return {
        'num_classes': len(class_names),
        'num_images': len(annotation_files),
        'class_names': list(class_names)
    }

def validate_training_config(parameters):
    """
    Validate training configuration parameters.
    
    Args:
        parameters: Dict containing training parameters
        
    Returns:
        str: Error message if validation fails, None otherwise
    """
    # Validate model type
    model_type = parameters.get('model_type')
    if not model_type:
        return "Model type is required"
    
    if model_type not in ['yolo', 'rf-detr']:
        return f"Unsupported model type: {model_type}"
    
    # Validate model variant
    model_variant = parameters.get('model_variant')
    if not model_variant:
        return "Model variant is required"
    
    supported_models = app.config.get('SUPPORTED_MODELS', {})
    if model_type in supported_models and model_variant not in supported_models[model_type]:
        return f"Unsupported model variant: {model_variant} for {model_type}"
    
    # Validate numeric parameters
    try:
        epochs = int(parameters.get('epochs', 0))
        if epochs <= 0:
            return "Epochs must be a positive integer"
        
        batch_size = int(parameters.get('batch_size', 0))
        if batch_size <= 0:
            return "Batch size must be a positive integer"
        
        learning_rate = float(parameters.get('learning_rate', 0))
        if learning_rate <= 0:
            return "Learning rate must be a positive number"
        
        validation_split = float(parameters.get('validation_split', 0))
        if validation_split <= 0 or validation_split >= 1:
            return "Validation split must be between 0 and 1"
    
    except (ValueError, TypeError):
        return "Invalid numeric parameter values"
    
    # All checks passed
    return None
