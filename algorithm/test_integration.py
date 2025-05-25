#!/usr/bin/env python3
"""
Test integration script for the MuJoCo + Physics Simulation pipette system.
This script tests all components and provides examples of how to use the system.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not available. Running physics-only tests.")

from pipette_physics_simulation import PipettePhysicsSimulator, PipetteConfig, Particle
from integrated_pipette_environment import IntegratedPipetteEnv

class TestSuite:
    """Test suite for the pipette simulation system"""
    
    def __init__(self):
        self.xml_path = "particle_pipette_system.xml"  # Your XML file
        self.test_results = {}
        
    def run_all_tests(self):
        """Run all test cases"""
        print("=" * 60)
        print("PIPETTE SIMULATION SYSTEM - TEST SUITE")
        print("=" * 60)
        
        # Test 1: Physics Simulation Only
        print("\n1. Testing Physics Simulation (No MuJoCo)...")
        self.test_physics_simulation()
        
        # Test 2: MuJoCo Integration (if available)
        if MUJOCO_AVAILABLE and os.path.exists(self.xml_path):
            print("\n2. Testing MuJoCo Integration...")
            self.test_mujoco_integration()
        else:
            print("\n2. Skipping MuJoCo tests (not available or XML not found)")
        
        # Test 3: Particle Pickup Logic
        print("\n3. Testing Particle Pickup Logic...")
        self.test_particle_pickup()
        
        # Test 4: Reward System
        print("\n4. Testing Reward System...")
        self.test_reward_system()
        
        # Test 5: State Management
        print("\n5. Testing State Management...")
        self.test_state_management()
        
        # Test 6: Full Environment Test
        if MUJOCO_AVAILABLE and os.path.exists(self.xml_path):
            print("\n6. Testing Full Environment...")
            self.test_full_environment()
        
        # Print summary
        self.print_test_summary()
    
    def test_physics_simulation(self):
        """Test the physics simulation independently"""
        try:
            # Create physics simulator
            config = PipetteConfig()
            sim = PipettePhysicsSimulator(config)
            
            # Add test particles
            sim.add_particle(np.array([0.0, 0.0, 0.02]), 0)
            sim.add_particle(np.array([0.01, 0.01, 0.02]), 1)
            sim.add_particle(np.array([0.05, 0.05, 0.02]), 2)  # Far away
            
            print(f"   ✓ Created simulator with {len(sim.particles)} particles")
            
            # Test pipette positioning
            tip_pos = np.array([0.0, 0.0, 0.03])
            sim.update_pipette_state(tip_pos, 0.5, 0.01)  # Medium plunger depth
            
            state = sim.get_state_dict()
            print(f"   ✓ Pipette state: {state['pipette_state']}")
            print(f"   ✓ Particles in range: {state['nearby_particle_count']}")
            
            # Test aspiration
            sim.update_pipette_state(tip_pos, 0.8, 0.01)  # Deep plunger
            state = sim.get_state_dict()
            print(f"   ✓ After aspiration attempt: {len(sim.held_particles)} particles held")
            
            self.test_results['physics_simulation'] = True
            
        except Exception as e:
            print(f"   ✗ Physics simulation test failed: {e}")
            self.test_results['physics_simulation'] = False
    
    def test_mujoco_integration(self):
        """Test MuJoCo XML loading and basic functionality"""
        try:
            # Test XML loading
            if not os.path.exists(self.xml_path):
                raise FileNotFoundError(f"XML file not found: {self.xml_path}")
            
            # model = mujoco_py.load_model_from_path(self.xml_path)
            # sim = mujoco_py.MjSim(model)

            model = mujoco.MjModel.from_xml_path(xml_path)
            sim = mujoco.MjData(model)
            
            print(f"   ✓ Successfully loaded MuJoCo model from {self.xml_path}")
            print(f"   ✓ Model has {model.nbody} bodies, {model.njnt} joints")
            
            # Test joint access
            joint_names = ['machine_x', 'machine_y', 'machine_z', 'plunger_joint']
            for joint_name in joint_names:
                try:
                    joint_id = model.joint_name2id(joint_name)
                    print(f"   ✓ Found joint '{joint_name}' with ID {joint_id}")
                except:
                    print(f"   ✗ Joint '{joint_name}' not found")
                    raise
            
            # Test actuator access
            actuator_names = ['x_control', 'y_control', 'z_control', 'plunger_control']
            for actuator_name in actuator_names:
                try:
                    actuator_id = model.actuator_name2id(actuator_name)
                    print(f"   ✓ Found actuator '{actuator_name}' with ID {actuator_id}")
                except:
                    print(f"   ✗ Actuator '{actuator_name}' not found")
                    raise
            
            # Test simulation step
            sim.step()
            print("   ✓ MuJoCo simulation step successful")
            
            self.test_results['mujoco_integration'] = True
            
        except Exception as e:
            print(f"   ✗ MuJoCo integration test failed: {e}")
            self.test_results['mujoco_integration'] = False
    
    def test_particle_pickup(self):
        """Test particle pickup mechanics"""
        try:
            sim = PipettePhysicsSimulator()
            
            # Add particles at different distances
            sim.add_particle(np.array([0.0, 0.0, 0.02]), 0)      # Very close
            sim.add_particle(np.array([0.01, 0.0, 0.02]), 1)     # Close
            sim.add_particle(np.array([0.03, 0.0, 0.02]), 2)     # At range limit
            sim.add_particle(np.array([0.1, 0.0, 0.02]), 3)      # Too far
            
            # Position pipette near particles
            tip_pos = np.array([0.0, 0.0, 0.025])
            
            # Test no pickup when plunger is retracted
            sim.update_pipette_state(tip_pos, 0.0, 0.01)
            assert len(sim.held_particles) == 0, "Should not pick up particles when plunger retracted"
            print("   ✓ No pickup when plunger retracted")
            
            # Test pickup with plunger extended
            for _ in range(10):  # Multiple steps to allow pickup
                sim.update_pipette_state(tip_pos, 0.8, 0.01)
            
            pickup_count = len(sim.held_particles)
            print(f"   ✓ Picked up {pickup_count} particles with extended plunger")
            
            # Test capacity limit
            sim.config.max_capacity = 2
            sim.held_particles.clear()
            
            for _ in range(20):  # Many steps to try to pickup all
                sim.update_pipette_state(tip_pos, 0.8, 0.01)
            
            final_count = len(sim.held_particles)
            assert final_count <= 2, f"Exceeded capacity limit: {final_count}"
            print(f"   ✓ Capacity limit respected: {final_count} <= 2")
            
            self.test_results['particle_pickup'] = True
            
        except Exception as e:
            print(f"   ✗ Particle pickup test failed: {e}")
            self.test_results['particle_pickup'] = False
    
    def test_reward_system(self):
        """Test the reward calculation system"""
        try:
            sim = PipettePhysicsSimulator()
            sim.add_particle(np.array([0.0, 0.0, 0.02]), 0)
            
            # Test aspiration reward
            tip_pos = np.array([0.0, 0.0, 0.025])
            
            # Record initial state
            initial_aspirations = len(sim.aspiration_history)
            
            # Perform aspiration
            for _ in range(10):
                sim.update_pipette_state(tip_pos, 0.8, 0.01)
            
            rewards = sim.calculate_reward_components()
            print(f"   ✓ Reward components: {rewards}")
            
            # Check if aspiration was rewarded
            if len(sim.held_particles) > 0:
                assert rewards['aspiration_reward'] > 0, "Should reward successful aspiration"
                print("   ✓ Aspiration reward working")
            
            # Test dispensing reward
            if len(sim.held_particles) > 0:
                for _ in range(10):
                    sim.update_pipette_state(tip_pos, 0.1, 0.01)  # Retract plunger
                
                rewards = sim.calculate_reward_components()
                print(f"   ✓ Post-dispense rewards: {rewards}")
            
            self.test_results['reward_system'] = True
            
        except Exception as e:
            print(f"   ✗ Reward system test failed: {e}")
            self.test_results['reward_system'] = False
    
    def test_state_management(self):
        """Test state transitions and management"""
        try:
            sim = PipettePhysicsSimulator()
            sim.add_particle(np.array([0.0, 0.0, 0.02]), 0)
            
            tip_pos = np.array([0.0, 0.0, 0.025])
            
            # Test idle state
            sim.update_pipette_state(tip_pos, 0.0, 0.01)
            assert sim.pipette_state.value == 'idle', f"Expected idle, got {sim.pipette_state.value}"
            print("   ✓ Idle state detected correctly")
            
            # Test aspirating state
            sim.update_pipette_state(tip_pos, 0.5, 0.01)
            # Simulate plunger movement
            sim.plunger_velocity = -0.2  # Moving up
            sim._update_pipette_state()
            print(f"   ✓ State after plunger up: {sim.pipette_state.value}")
            
            # Test state transitions
            states_observed = set()
            for depth in [0.0, 0.2, 0.5, 0.8, 0.5, 0.2, 0.0]:
                sim.update_pipette_state(tip_pos, depth, 0.01)
                states_observed.add(sim.pipette_state.value)
            
            print(f"   ✓ Observed states: {states_observed}")
            
            self.test_results['state_management'] = True
            
        except Exception as e:
            print(f"   ✗ State management test failed: {e}")
            self.test_results['state_management'] = False
    
    def test_full_environment(self):
        """Test the full integrated environment"""
        try:
            env = IntegratedPipetteEnv(self.xml_path)
            
            # Test reset
            obs = env.reset()
            print(f"   ✓ Environment reset, observation shape: {obs.shape}")
            
            # Test action space
            action = env.action_space.sample()
            print(f"   ✓ Sample action: {action}")
            
            # Test step
            obs, reward, done, info = env.step(action)
            print(f"   ✓ Step completed: reward={reward:.3f}, done={done}")
            print(f"   ✓ Info keys: {list(info.keys())}")
            
            # Test multiple steps
            total_reward = 0
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            print(f"   ✓ Completed {i+1} steps, total reward: {total_reward:.3f}")
            
            # Test render (if possible)
            try:
                env.render()
                print("   ✓ Rendering successful")
            except:
                print("   ! Rendering not available (no display)")
            
            env.close()
            self.test_results['full_environment'] = True
            
        except Exception as e:
            print(f"   ✗ Full environment test failed: {e}")
            self.test_results['full_environment'] = False
    
    def print_test_summary(self):
        """Print summary of all test results"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:.<30} {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! System ready for use.")
        else:
            print("⚠️  Some tests failed. Check the errors above.")

def create_sample_xml():
    """Create a minimal XML file for testing if the original is not available"""
    xml_content = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_pipette">
    <worldbody>
        <body name="pipette_base" pos="0 0 0.2">
            <joint name="machine_x" type="slide" axis="1 0 0" range="-0.4 0.4"/>
            <joint name="machine_y" type="slide" axis="0 1 0" range="-0.4 0.4"/>
            <inertial pos="0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1"/>
            
            <body name="pipette_body" pos="0 0 0">
                <joint name="machine_z" type="slide" axis="0 0 1" range="-0.3 0.1"/>
                <inertial pos="0 0 -0.08" mass="0.8" diaginertia="0.05 0.05 0.08"/>
                <geom name="pipette_wall" type="box" size="0.015 0.015 0.08" pos="0 0 -0.05" mass="0.1"/>
                
                <body name="plunger" pos="0 0 0.03">
                    <joint name="plunger_joint" type="slide" axis="0 0 1" range="0 0.04"/>
                    <geom name="plunger_head" type="box" size="0.015 0.015 0.015" pos="0 0 0" mass="0.1"/>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <position name="x_control" joint="machine_x" ctrlrange="-0.4 0.4" kp="100"/>
        <position name="y_control" joint="machine_y" ctrlrange="-0.4 0.4" kp="100"/>
        <position name="z_control" joint="machine_z" ctrlrange="-0.3 0.1" kp="100"/>
        <position name="plunger_control" joint="plunger_joint" ctrlrange="0 0.04" kp="50"/>
    </actuator>
</mujoco>'''
    
    with open("test_pipette.xml", "w") as f:
        f.write(xml_content)
    
    return "test_pipette.xml"

def interactive_demo():
    """Run an interactive demo of the system"""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO")
    print("=" * 60)
    
    try:
        # Try to use the full environment
        if MUJOCO_AVAILABLE:
            xml_path = "simulation/particle_pipette_system.xml"
            if not os.path.exists(xml_path):
                print("Main XML not found, creating test XML...")
                xml_path = create_sample_xml()
            
            env = IntegratedPipetteEnv(xml_path)
            print("✓ Full environment loaded")
        else:
            print("MuJoCo not available, using physics simulation only")
            env = None
        
        # Physics-only demo
        sim = PipettePhysicsSimulator()
        
        # Add particles in a realistic configuration
        sim.add_particle(np.array([-0.3, -0.3, 0.02]), 0, well_id=1)
        sim.add_particle(np.array([-0.29, -0.3, 0.02]), 1, well_id=1)
        sim.add_particle(np.array([-0.3, -0.29, 0.02]), 2, well_id=1)
        
        print(f"Added {len(sim.particles)} particles to simulation")
        
        # Simulate a pipetting sequence
        print("\nSimulating pipetting sequence...")
        
        # Move to well 1
        tip_pos = np.array([-0.3, -0.3, 0.03])
        print("1. Moving to well 1...")
        
        # Lower pipette
        for i in range(10):
            z_pos = 0.03 - (i * 0.001)  # Gradually lower
            sim.update_pipette_state(np.array([-0.3, -0.3, z_pos]), 0.0, 0.01)
        
        print("2. Lowering pipette...")
        
        # Aspirate
        print("3. Aspirating...")
        for i in range(20):
            plunger_depth = min(0.8, i * 0.04)  # Gradually extend plunger
            sim.update_pipette_state(tip_pos, plunger_depth, 0.01)
            
            if i % 5 == 0:
                state = sim.get_state_dict()
                print(f"   Step {i}: {state['held_particle_count']} particles held, state: {state['pipette_state']}")
        
        # Move to well 3
        print("4. Moving to well 3...")
        target_pos = np.array([0.3, -0.3, 0.03])
        for i in range(20):
            progress = i / 19.0
            current_pos = tip_pos * (1 - progress) + target_pos * progress
            sim.update_pipette_state(current_pos, 0.8, 0.01)
        
        # Dispense
        print("5. Dispensing...")
        for i in range(20):
            plunger_depth = 0.8 - (i * 0.04)  # Gradually retract plunger
            sim.update_pipette_state(target_pos, max(0.0, plunger_depth), 0.01)
            
            if i % 5 == 0:
                state = sim.get_state_dict()
                print(f"   Step {i}: {state['held_particle_count']} particles held, state: {state['pipette_state']}")
        
        # Final state
        final_state = sim.get_state_dict()
        print(f"\nFinal state:")
        print(f"  Particles held: {final_state['held_particle_count']}")
        print(f"  Aspirations: {len(sim.aspiration_history)}")
        print(f"  Dispensed: {len(sim.dispense_history)}")
        
        # Calculate rewards
        rewards = sim.calculate_reward_components()
        total_reward = sum(rewards.values())
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Reward breakdown: {rewards}")
        
        # Clean up
        if env:
            env.close()
        
        print("\n✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        interactive_demo()
    else:
        # Run full test suite
        test_suite = TestSuite()
        test_suite.run_all_tests()
        
        # Ask if user wants to run demo
        try:
            if input("\nRun interactive demo? (y/n): ").lower().startswith('y'):
                interactive_demo()
        except KeyboardInterrupt:
            print("\nExiting...")