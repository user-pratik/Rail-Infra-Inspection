import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from transformers import Dinov2ForImageClassification
import onnxruntime
import numpy as np
import time
import pandas as pd
from PIL import Image

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


class BridgeCSVDataset(Dataset):
    """Custom dataset loader for Roboflow _classes.csv schema"""
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        csv_path = os.path.join(data_dir, "_classes.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            class_cols = [c for c in df.columns if c != "filename"]
            for _, row in df.iterrows():
                img_path = os.path.join(data_dir, row["filename"])
                if os.path.exists(img_path):
                    one_hot = row[class_cols].values.astype(np.float32)
                    label = int(np.argmax(one_hot)) if np.sum(one_hot) > 0 else 0
                    self.samples.append((img_path, label))
        else:
            for fname in os.listdir(data_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(data_dir, fname), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ModelProfiler:
    def __init__(self, test_data_path: str, batch_size: int = 1):
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_data()
        
    def setup_data(self):
        """Setup data transformations and loader"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Check if directory has subfolders (ImageFolder) or flat files with _classes.csv
        csv_file = os.path.join(self.test_data_path, "_classes.csv")
        subdirs = [d for d in os.listdir(self.test_data_path) if os.path.isdir(os.path.join(self.test_data_path, d))]
        
        if os.path.exists(csv_file) or len(subdirs) == 0:
            self.test_dataset = BridgeCSVDataset(self.test_data_path, transform)
        else:
            self.test_dataset = datasets.ImageFolder(self.test_data_path, transform)
            
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False
        )
        
    def profile_pytorch_model(self, model_path):
        """Profile PyTorch DINOV2 model"""
        model = Dinov2ForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=5
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        print("Started profiling PyTorch model...")
        
        latencies = []
        accuracies = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                start_time = time.time()
                outputs = model(images).logits
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # ms
                preds = torch.argmax(outputs, dim=1)
                accuracies.append((preds == labels).float().mean().item())

            print(f"PyTorch Accuracy: {np.mean(accuracies) * 100:.2f}%, Avg Latency: {np.mean(latencies):.2f} ms")
        
        return {
            "framework": "PyTorch",
            "avg_latency": np.mean(latencies),
            "accuracy": np.mean(accuracies) * 100
        }
        
    def profile_onnx_model(self, model_path):
        """Profile ONNX model"""
        if not os.path.exists(model_path):
            print(f"ONNX model not found at {model_path}, skipping.")
            return {"framework": "ONNX", "avg_latency": None, "accuracy": None}

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        latencies = []
        accuracies = []

        print("Started profiling ONNX model...")
        
        for images, labels in self.test_loader:
            images_np = images.numpy()
            
            start_time = time.time()
            outputs = session.run(None, {input_name: images_np})
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)
            preds = np.argmax(outputs[0], axis=1)
            accuracies.append((preds == labels.numpy()).mean())
        print(f"ONNX Accuracy: {np.mean(accuracies) * 100:.2f}%, Avg Latency: {np.mean(latencies):.2f} ms")
            
        return {
            "framework": "ONNX",
            "avg_latency": np.mean(latencies),
            "accuracy": np.mean(accuracies) * 100
        }
        
    def profile_tensorrt_model(self, engine_path):
        if not HAS_TENSORRT or not os.path.exists(engine_path):
            print(f"TensorRT engine not found or TensorRT runtime unavailable, skipping TensorRT.")
            return {
                "framework": "TensorRT",
                "avg_latency": None,
                "accuracy": None
            }

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        context = engine.create_execution_context()
        
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        
        latencies = []
        accuracies = []
        
        for images, labels in self.test_loader:
            input_batch = images.numpy()
            d_input = cuda.mem_alloc(input_batch.nbytes)
            output = np.empty((self.batch_size, 5), dtype=np.float32)
            d_output = cuda.mem_alloc(output.nbytes)
            
            stream = cuda.Stream()
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
            
            start_time = time.time()
            cuda.memcpy_htod_async(d_input, input_batch, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)
            preds = np.argmax(output, axis=1)
            accuracies.append((preds == labels.numpy()).mean())
            
            d_input.free()
            d_output.free()
        
        print(f"TensorRT Accuracy: {np.mean(accuracies) * 100:.2f}%, Avg Latency: {np.mean(latencies):.2f} ms")
            
        return {
            "framework": "TensorRT",
            "avg_latency": np.mean(latencies),
            "accuracy": np.mean(accuracies) * 100
        }

    def run_complete_profile(self, pytorch_path, onnx_path, tensorrt_path=None):
        """Run complete profiling"""
        results = []
        
        # Profile PyTorch
        if os.path.exists(pytorch_path):
            results.append(self.profile_pytorch_model(pytorch_path))
        
        # Profile ONNX
        if onnx_path and os.path.exists(onnx_path):
            results.append(self.profile_onnx_model(onnx_path))
        
        # Profile TensorRT
        if tensorrt_path and os.path.exists(tensorrt_path) and HAS_TENSORRT:
            results.append(self.profile_tensorrt_model(tensorrt_path))
        
        df = pd.DataFrame(results)
        return df