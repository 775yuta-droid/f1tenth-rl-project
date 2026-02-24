# F1Tenth RL Project

## Project Overview
This project focuses on combining reinforcement learning techniques with autonomous driving in the F1Tenth racing environment. The goal is to train agents that can navigate a race track autonomously, using various machine learning algorithms and simulation environments.

## Hardware Requirements
- **Raspberry Pi 4 Model B**
- **Compatible Cameras** (USB or Raspberry Pi camera)
- **Motor Driver** (for controlling motors)
- **Power Supply**
- **Chassis** (to assemble the hardware components)

## Learning Time Estimates
- **Basic Understanding of Python:** 2-4 weeks
- **Introduction to Reinforcement Learning:** 3-6 weeks
- **Familiarization with the F1Tenth Environment:** 1-2 weeks

## Quick Start Guide
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/775yuta-droid/f1tenth-rl-project.git
   cd f1tenth-rl-project
   ```
2. **Set Up Your Hardware:** Follow the hardware setup guide provided in the repository.
3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Simulation:**
   ```bash
   python main.py
   ```

## config.py Explanations
- **API Keys:** Store API keys needed for connecting to external services.
- **Model Parameters:** Adjust hyperparameters for training the reinforcement learning models.
- **Environment Settings:** Configure the simulation environment settings for testing.

## Troubleshooting
- **Camera Issues:** Ensure the camera is properly connected and configured in the settings.
- **Performance Problems:** Check the hardware specifications and optimize the code for better performance.

## Implementation References
- **Reinforcement Learning Resources:**
  - Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction."
  - Various online courses on Coursera and Udacity about RL.
- **F1Tenth Resources:**
  - F1Tenth Autonomous Racing GitHub repository.
  - Research papers on autonomous driving and racing algorithms.

## Conclusion
This project serves as a comprehensive guide to implementing reinforcement learning for autonomous racing. It combines practical hardware setups with theoretical learning, making it ideal for students and enthusiasts in the field.