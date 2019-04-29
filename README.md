# Gazebo-OpenAIGym
Minimal OpenAI gym interface for Gazebo Robots

This repository takes inspiration from the now deprecated gym-gazebo (https://github.com/erlerobot/gym-gazebo).
However, this implementation is much more lightweight, facilitating easy usage with the Turtlebot like robots.  

We also include an implementation of the Dueling DQN [1] RL agent with this repository for an user to get started. 

This has been tested with ROS Kinetic, Ubuntu 16.04. 

For the DeepRL agent, we use PyTorch 1.0.1. 



Instructions - 1) Clone the repository
               2) For intializing an environment, pass the location of the launch file when creating the environment. We 
                  include an example world file in the /worlds directory.
               3) Run rllearn.py with the required dependencies to begin learning on your gazebo world. Change the reward                         functions in the TurtlebotGym.py file to suit your needs. 


