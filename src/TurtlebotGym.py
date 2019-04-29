#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import signal
import subprocess
import time
from std_srvs.srv import Empty
import random
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty as Empty_msg
#from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
import numpy as np

class TurtlebotGym:
    
    def __init__(self, launchfile='/opt/ros/kinetic/share/turtlebot_gazebo/launch/turtlebot_world.launch'):
        
        self.gazebo_factor = 1 #simulation speed factor, there are some issues with not setting as 1
        self.time_step = 0.25 #250ms time step for RL
        self.max_vel = 0.3
        self.num_actions=3 #Forward, Left, Right
        self.stopping_distance=0.5 #distance from obstacle at which we consider it to be a collision
        self.goal = np.array([3,0]) #goal state for the RL agent
        self.goal_dist = 0.1 #how close we get to goal state before success
        self.max_timesteps=100 #max number of timesteps per episode
        self.render=True
        self.sticky_actions=5 #number of start sticky actions for random start
        

        #random_number = random.randint(10000, 15000)
        #self.port = str(random_number) #os.environ["ROS_PORT_SIM"]
        #self.port_gazebo = str(random_number+1) #os.environ["ROS_PORT_SIM"]

        self.port=str(11311)
        self.port_gazebo=str(11312)

        os.environ["ROS_MASTER_URI"] = "http://localhost:"+self.port
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:"+self.port_gazebo
       
        print("ROS_MASTER_URI=http://localhost:"+self.port + "\n")
        print("GAZEBO_MASTER_URI=http://localhost:"+self.port_gazebo + "\n")

        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        if not os.path.exists(launchfile):
            raise IOError("File "+launchfile+" does not exist")

        self._roslaunch = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", self.port, launchfile])
        print ("Gazebo launched, waiting for 3 seconds!")
        
        if(not(self.render)):
            os.system("killall -9 gzclient")
        
        time.sleep(3)

        self.gzclient_pid = 0

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym')
        
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
        self.reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty_msg,queue_size=1) #odometry reset
         
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty) #pause physics
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty) #reset world, sim time doesnt reset!
         
    def step(self, action):
      
        #step method
        self.elapsed_time=self.elapsed_time + 1 #timestep count of the simulator
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.last_timestep = rospy.get_time() #we do this to make sure simulator runs for time_step duration
        
        if action == 1: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = self.max_vel
            vel_cmd.angular.z = 0.0
            
        elif action == 0: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = self.max_vel
            
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = self.max_vel
            vel_cmd.angular.z = -self.max_vel
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        self.vel_pub.publish(vel_cmd)
        
        laser_data = None
        odom_data= None
        #while data is None:
        try:
            laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            odom_data  = rospy.wait_for_message('/odom',Odometry,timeout=5)
        except:
            print('NO LASER or ODOM')
            pass
        #print('GOT READINGS')
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #we sleep till timestep duration simulation is done
            time.sleep(max(0,(self.last_timestep+self.time_step - rospy.get_time()))/self.gazebo_factor) #sleep if needed
            #print("timestep length was {}".format(rospy.get_time()-self.last_timestep))
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        return self.make_state(laser_data,odom_data,action)
        
    def reset(self):

        self.elapsed_time=0
        rospy.wait_for_service('/gazebo/reset_world')
        for t in range(self.sticky_actions):
            self.step(random.randint(0,self.num_actions-1))
        try:
            self.reset_proxy()
            self.reset_odom.publish(Empty_msg()) #reset odom frame at the start of the course
            self.start_time=rospy.get_time()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        
        laser_data = None
        odom_data= np.array([0,0]) #start, origin is zero
        #while data is None:
        try:
            laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            odom_data  = rospy.wait_for_message('/odom',Odometry,timeout=5)
        except:
            print('NO LASER')
            pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
            #print('SIMULATION PAUSED')
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        return self.make_state(laser_data,odom_data)

    def render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        else:
            self.gzclient_pid = 0
            
    def make_state(self,laser_data,odom_data,action=None):
        
        done = 0
        state=np.array([0,0,0]) #by default
        crashed=False
        
        if odom_data is not None:
            x=odom_data.pose.pose.position.x
            y=odom_data.pose.pose.position.y
            angle = odom_data.pose.pose.orientation.z

            state = np.array([x,y,angle]) #x,y,theta
            
            #Things from here is SPECIFIC to your environment. Choose the reward appropriately
            if x<2:
                reward = -np.absolute(y)
            else:
                reward = -np.absolute(x-2.5)
            
            if crashed:
                reward=-10
            elif np.linalg.norm(state[:-1]-self.goal,2)<self.goal_dist:
                done=1
                reward=10
            elif (np.absolute(state[:-1])>3).any():
                done=1
                reward=-10
                crashed=True
            #SPECIFIC CODE OVER
                
        if self.elapsed_time>self.max_timesteps:
            done=1
            
        if action is None:
            return state #called from reset function
        else:
            return state, reward, done, crashed #called from step function

    def close(self):

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
            os.wait()