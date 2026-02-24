# F1Tenth RL Project

## Overview
This project implements reinforcement learning algorithms for autonomous racing in a simulated environment.

## Hardware Requirements
- **Computer**: A machine running Ubuntu 18.04 or later with a multi-core processor.
- **RAM**: At least 16GB of RAM.
- **GPU**: NVIDIA GPU recommended for improved performance, with CUDA support.
- **Camera**: A webcam for real-time image processing (optional).

## Learning Time Estimates
| Task                      | Estimated Time |
|---------------------------|----------------|
| Setting up the environment | 2-4 hours      |
| Understanding the code    | 1-3 days       |
| Training the model        | 1-2 weeks      |
| Testing & validation      | 3-5 days       |

## Config.py Explanations
- **learning_rate**: Determines how quickly the model learns.
- **batch_size**: Number of samples processed before the model updates.
- **num_episodes**: Total episodes to train the model. 

## Troubleshooting
### Common Issues & Diagnostic Commands
- **Environment Issues**:
    - Check if all dependencies are installed:
      ```bash
      pip list
      ```
    - Verify CUDA installation:
      ```bash
      nvcc --version
      ```
- **Code Errors**:
    - Run linter to find issues:
      ```bash
      flake8 .
      ```

## Implementation References
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## Contribution Guidelines
- Fork the repository and create a new branch for your feature/bugfix.
- Ensure that your code passes the existing tests and add new tests for your changes.
- Submit a pull request with a clear description of your changes. 
  
Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more details.