import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
import argparse

class OnnxablePolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # NOTE: SB3 policy.predict() uses a different internal flow, 
        # but the core actor is what we need for deployment.
        # Observation shape: [Batch, 110]
        return self.policy(observation)

def convert_to_onnx(model_path, onnx_path):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    # Policy extraction
    # model.policy is the ActorCriticPolicy
    # We want the 'actor' part for inference
    
    class PPOInferenceModel(nn.Module):
        def __init__(self, policy, low, high):
            super().__init__()
            self.policy = policy
            self.register_buffer("low", torch.tensor(low, dtype=torch.float32))
            self.register_buffer("high", torch.tensor(high, dtype=torch.float32))
            
        def forward(self, obs):
            # Deterministic actions (mean of the distribution)
            actions = self.policy.get_distribution(obs).get_actions(deterministic=True)
            # Clip actions to space limits
            return torch.clamp(actions, self.low, self.high)

    low = model.action_space.low
    high = model.action_space.high
    inference_model = PPOInferenceModel(model.policy, low, high)
    inference_model.eval()

    # Dummy input based on RLDriver logic (108 lidar + 2 state = 110)
    # Check actual observation space if possible
    obs_dim = model.observation_space.shape[0]
    print(f"Detected observation dimension: {obs_dim}")
    
    dummy_input = torch.randn(1, obs_dim)

    print(f"Exporting to ONNX: {onnx_path}...")
    torch.onnx.export(
        inference_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model")
    parser.add_argument("--output", type=str, help="Output path for .onnx model")
    args = parser.parse_args()

    if not args.output:
        args.output = args.model.replace(".zip", ".onnx")

    convert_to_onnx(args.model, args.output)
