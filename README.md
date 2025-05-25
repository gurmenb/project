pip install -r requirements.txt

# env

python3 test_pipette_env.py

python train_pipette_agent.py

# to run the mujoco simulator

go into mujoco-3.3.2/bin

run: ./simulate ../model/pipette/pipette_system.xml
./simulate ../model/universal_robots_ur5e/ur5e.xml

# Test everything (this will verify your XML file works)

python3 test_integration.py

# Run interactive demo to see it in action

python3 test_integration.py demo
