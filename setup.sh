#Install tensorflow
pip3 install tensorflow

#Install rllab
git clone https://github.com/rll/rllab.git

#Setup rllab for linux
/bin/bash rllab/scripts/setup_linux.sh

#Download mujoco zip
wget https://roboti.us/download/mjpro131_linux.zip

#Setup mujoco for linux, you will have to specify the mujoco zip file and the mujoco key file
/bin/bash rllab/scripts/setup_mujoco.sh
