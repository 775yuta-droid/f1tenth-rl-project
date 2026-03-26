import onnxruntime as ort
import torch
import numpy as np
from stable_baselines3 import PPO
import argparse

def verify_conversion(sb3_path, onnx_path):
    print(f"Verifying {sb3_path} vs {onnx_path}...")
    
    # 1. Load SB3 model
    sb3_model = PPO.load(sb3_path, device="cpu")
    
    # 2. Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # 3. Create random input
    obs_dim = sb3_model.observation_space.shape[0]
    dummy_input = np.random.randn(1, obs_dim).astype(np.float32)
    
    # 4. SB3 Inference (deterministic)
    # predict() returns (actions, next_state)
    sb3_action, _ = sb3_model.predict(dummy_input, deterministic=True)
    
    # 5. ONNX Inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_action = ort_outputs[0]
    
    print(f"SB3 Action Shape : {sb3_action.shape}")
    print(f"ONNX Action Shape: {onnx_action.shape}")
    
    print(f"SB3 Action  : {sb3_action}")
    print(f"ONNX Action : {onnx_action}")
    
    diff = np.abs(sb3_action - onnx_action)
    print(f"Max difference: {np.max(diff)}")
    
    if np.allclose(sb3_action, onnx_action, atol=1e-5):
        print("Success: ONNX outputs match SB3 outputs!")
    else:
        print("Warning: Outputs differ significantly!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sb3", type=str, required=True, help="Path to SB3 .zip model")
    parser.add_argument("--onnx", type=str, help="Path to .onnx model")
    args = parser.parse_args()
    
    if not args.onnx:
        args.onnx = args.sb3.replace(".zip", ".onnx")
        
    verify_conversion(args.sb3, args.onnx)
