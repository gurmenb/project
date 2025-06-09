Setup:
pip install -r requirements.txt

Run these commands to generate new runs:
python3 train.py

additional commands to run:
python3 train.py --config configs/your_config_here.yaml --name "your_name_here"

To view tensorboard logs:
in the main project directory run:
tensorboard --logdir logs
Then go to: http://localhost:6006
