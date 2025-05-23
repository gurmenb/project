pip install -r requirements.txt

# env

python3 test_pipette_env.py

python train_pipette_agent.py

# to run the mujoco simulator

go into mujoco-3.3.2/bin
give yourself
run: ./simulate ../model/pipette/pipette_system.xml
./simulate ../model/universal_robots_ur5e/ur5e.xml
