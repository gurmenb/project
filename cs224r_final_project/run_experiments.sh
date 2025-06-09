bash
#!/bin/bash

echo "Starting PPO Experiments..."

# Entropy Experiments
echo "Running entropy experiments..."
python train.py --config configs/config_entropy_005.yaml --name "entropy_005_low_"
python train.py --config configs/config_entropy_000.yaml --name "entropy_000_none_"
python train.py --config configs/config_entropy_020.yaml --name "entropy_020_high_"

python train.py  --name "baseline"
python train.py  --name "target_wrong_pos-change-8-12"
python train.py  --name "target_pos-change-10-8"
python train.py  --name "target_pos-change-5-10"
python train.py  --name "source_pos-change-5-10"
python3 train.py  --name "source_pos-change-10-2"
python3 train.py --config configs/config_clip_05.yaml --name "clip_05_very_conserv"
python3 train.py --config configs/config_clip_5.yaml --name "clip_5_very_aggressive"

python3 train.py --config configs/config_radius_half_0.5.yaml --name "new_radius_half_0.5"
python3 train.py --config configs/config_radius_quarter_0.25.yaml --name "new_radius_quarter_0.25"

# Clip Ratio Experiments  
echo "Running clip ratio experiments..."
python train.py --config configs/config_clip_010.yaml --name "clip_010_conservative"
# python train.py --config config_clip_015.yaml --name "clip_015_mild_conserv"
python train.py --config configs/config_clip_030.yaml --name "clip_030_aggressive"
# python train.py --config config_clip_050.yaml --name "clip_050_very_aggressive"

echo "All experiments complete!"


### *Run All Experiments:*
bash
chmod +x run_experiments.sh
./run_experiments.sh
