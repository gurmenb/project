import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple
import mujoco_py

from pipette_physics_simulation import PipettePhysicsSimulator, PipetteConfig, PipetteEnvironmentWrapper

class IntegratedPipetteEnv(gym.Env):
    """
    Integrated environment that combines MuJoCo visualization with
    physics-based particle aspiration simulation for actor-critic training.
    """

    def __init__(self, mujoco_model_path: str = "particle_pipette_system.xml"):
        super().__init__()

        # 1) Load MuJoCo
        self.model = mujoco_py.load_model_from_path(mujoco_model_path)
        self.sim   = mujoco_py.MjSim(self.model)
        self.data  = self.sim.data
        self.viewer = None

        # 2) Initialize physics‐only simulator + wrapper
        config = PipetteConfig(
            max_capacity       = 3,
            suction_range      = 0.03,
            min_suction_depth  = 0.02,
            max_suction_depth  = 0.08,
            suction_force_max  = 8.0
        )
        self.physics_sim = PipettePhysicsSimulator(config)
        self.env_wrapper = PipetteEnvironmentWrapper(self.physics_sim)

        # 3) Define action space: normalized [-1,1]^4
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
            dtype= np.float32
        )

        # 4) Define observation space: 26-dimensional float32
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (26,),
            dtype = np.float32
        )

        # 5) Precompute MuJoCo joint/actuator indices & DOF addresses
        self._setup_mujoco_indices()

        # 6) Spawn particles in physics_sim (clears old)
        self._initialize_particles()

        # 7) Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 500

    def _setup_mujoco_indices(self):
        """Find each joint’s qpos/ctrl indices and each particle’s body‐joint address."""
        # Joint IDs
        self.joint_id = {
            'x': self.model.joint_name2id('machine_x'),
            'y': self.model.joint_name2id('machine_y'),
            'z': self.model.joint_name2id('machine_z'),
            'plunger': self.model.joint_name2id('plunger_joint')
        }
        # qpos address & qvel address for each
        self.joint_qposadr = {
            name: self.model.joint_qposadr[jid]
            for name, jid in self.joint_id.items()
        }
        self.joint_dofadr = {
            name: self.model.joint_dofadr[jid]
            for name, jid in self.joint_id.items()
        }

        # Actuator IDs
        self.actuator_id = {
            'x': self.model.actuator_name2id('x_control'),
            'y': self.model.actuator_name2id('y_control'),
            'z': self.model.actuator_name2id('z_control'),
            'plunger': self.model.actuator_name2id('plunger_control')
        }

        # Map each particle’s MuJoCo body name → body ID
        self.particle_body_id = {}
        for i in range(1, 7):  # well_1: particle_1_1 … particle_1_6
            try:
                bid = self.model.body_name2id(f'particle_1_{i}')
                self.particle_body_id[f'particle_1_{i}'] = bid
            except KeyError:
                pass
        for i in range(1, 5):  # well_2: particle_2_1 … particle_2_4
            try:
                bid = self.model.body_name2id(f'particle_2_{i}')
                self.particle_body_id[f'particle_2_{i}'] = bid
            except KeyError:
                pass
        for i in range(1, 6):  # loose: loose_particle_1 … loose_particle_5
            try:
                bid = self.model.body_name2id(f'loose_particle_{i}')
                self.particle_body_id[f'loose_particle_{i}'] = bid
            except KeyError:
                pass

        # Each particle body has a freejoint ⇒ 7 DOFs; record qpos address for each
        self.body_jntadr = {
            name: self.model.body_jntadr[bid]
            for name,bid in self.particle_body_id.items()
        }

    def _initialize_particles(self):
        """
        Reset physics_sim and respawn all particles at the positions
        specified in the XML, so MuJoCo shows them in the same place.
        """
        self.physics_sim.reset()
        idx = 0

        # Well 1 center = (-0.12, 0, 0.017)
        w1 = np.array([-0.12, 0.0, 0.017], dtype=np.float32)
        offs_w1 = [
            np.array([-0.02, -0.015, 0.0], dtype=np.float32),
            np.array([ 0.02, -0.015, 0.0], dtype=np.float32),
            np.array([-0.02,  0.015, 0.0], dtype=np.float32),
            np.array([ 0.02,  0.015, 0.0], dtype=np.float32),
            np.array([ 0.0,   0.0,   0.0], dtype=np.float32),
            np.array([ 0.0,   0.0,   0.008], dtype=np.float32)
        ]
        for off in offs_w1:
            self.physics_sim.add_particle(w1 + off, idx, well_id=1)
            idx += 1

        # Well 2 center = (0, 0, 0.017)
        w2 = np.array([0.0, 0.0, 0.017], dtype=np.float32)
        offs_w2 = [
            np.array([-0.015, -0.015, 0.0], dtype=np.float32),
            np.array([ 0.015, -0.015, 0.0], dtype=np.float32),
            np.array([-0.015,  0.015, 0.0], dtype=np.float32),
            np.array([ 0.015,  0.015, 0.0], dtype=np.float32)
        ]
        for off in offs_w2:
            self.physics_sim.add_particle(w2 + off, idx, well_id=2)
            idx += 1

        # Container center = (0, 0.1, 0.025)
        ctr = np.array([0.0, 0.1, 0.025], dtype=np.float32)
        offs_ctr = [
            np.array([-0.015, -0.015, 0.0], dtype=np.float32),
            np.array([ 0.015, -0.015, 0.0], dtype=np.float32),
            np.array([-0.015,  0.015, 0.0], dtype=np.float32),
            np.array([ 0.015,  0.015, 0.0], dtype=np.float32),
            np.array([ 0.0,    0.0,   0.01], dtype=np.float32)
        ]
        for off in offs_ctr:
            self.physics_sim.add_particle(ctr + off, idx, well_id=None)
            idx += 1

    def _get_mujoco_body_name(self, pidx: int) -> str:
        """Map physics particle index → MuJoCo body name."""
        if pidx < 6:
            return f'particle_1_{pidx+1}'
        elif pidx < 10:
            return f'particle_2_{pidx-5}'
        else:
            return f'loose_particle_{pidx-9}'

    def _sync_particles_to_mujoco(self):
        """
        Copy each free particle’s position & velocity back into MuJoCo
        qpos/qvel, so the viewer shows them in the correct place.
        """
        for i,p in enumerate(self.physics_sim.particles):
            body_name = self._get_mujoco_body_name(i)
            if body_name not in self.body_jntadr:
                continue
            jntadr = self.body_jntadr[body_name]
            if not p.is_held:
                # qpos[jntadr : jntadr+3] ← position
                self.sim.data.qpos[jntadr : jntadr+3] = p.position
                # qvel[jntadr : jntadr+3] ← velocity
                self.sim.data.qvel[jntadr : jntadr+3] = p.velocity

    def _get_particle_distance_features(self, physics_state: Dict, max_particles: int) -> np.ndarray:
        """
        Return an array of length `max_particles`: the distances of the nearest particles
        to the pipette tip. Pad with zeros if fewer than `max_particles` in range.
        """
        particles_in_range = physics_state.get('particles_in_range', [])
        feats = np.zeros(max_particles, dtype=np.float32)
        for i, pinfo in enumerate(particles_in_range[:max_particles]):
            feats[i] = float(pinfo['distance'])
        return feats

    def _get_observation(self) -> np.ndarray:
        """
        Construct the 26‐dimensional observation:
          (1) 4 MuJoCo joint positions [x,y,z,plunger]
          (2) 4 MuJoCo joint velocities
          (3) pipette tip coords (physics)      → 3 dims
          (4) plunger position (physics)       → 1 dim
          (5) held_particle_count              → 1 dim
          (6) nearby_particle_count            → 1 dim
          (7) suction_pressure                 → 1 dim
          (8) one‐hot pipette_state flags      → 3 dims
              [aspirating, dispensing, holding]
          (9) nearest‐particle distances (up to 8) → 8 dims
        """
        # 1–2) MuJoCo joint pos & vel
        jp = np.array([
            self.sim.data.qpos[self.joint_qposadr['x']],
            self.sim.data.qpos[self.joint_qposadr['y']],
            self.sim.data.qpos[self.joint_qposadr['z']],
            self.sim.data.qpos[self.joint_qposadr['plunger']]
        ], dtype=np.float32)

        jv = np.array([
            self.sim.data.qvel[self.joint_dofadr['x']],
            self.sim.data.qvel[self.joint_dofadr['y']],
            self.sim.data.qvel[self.joint_dofadr['z']],
            self.sim.data.qvel[self.joint_dofadr['plunger']]
        ], dtype=np.float32)

        # 3–7) physics state
        physics_state = self.physics_sim.get_state_dict()
        tip_pos   = np.array(physics_state['pipette_tip_position'], dtype=np.float32)   # (3,)
        pl_pos    = np.array([physics_state['plunger_position']], dtype=np.float32)      # (1,)
        held_cnt  = np.array([physics_state['held_particle_count']], dtype=np.float32)   # (1,)
        near_cnt  = np.array([physics_state['nearby_particle_count']], dtype=np.float32) # (1,)
        suction_p = np.array([physics_state['suction_pressure']], dtype=np.float32)      # (1,)

        # pipette_state one-hot
        ps = physics_state['pipette_state']
        f_asp  = np.array([float(ps == 'aspirating')], dtype=np.float32)
        f_disp = np.array([float(ps == 'dispensing')], dtype=np.float32)
        f_hold = np.array([float(ps == 'holding')], dtype=np.float32)

        # 9) nearest‐particle distances
        dist_feats = self._get_particle_distance_features(physics_state, 8)  # (8,)

        obs = np.concatenate([
            jp,           # 4
            jv,           # 4
            tip_pos,      # 3
            pl_pos,       # 1
            held_cnt,     # 1
            near_cnt,     # 1
            suction_p,    # 1
            f_asp, f_disp, f_hold,  # 3
            dist_feats    # 8
        ]).astype(np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray,float,bool,Dict]:
        """
        1) Scale normalized action [-1,1]^4 → MuJoCo ctrl ranges
           x: [-0.5, 0.5], y: [-0.5, 0.5], z: [0,1], plunger: [0,0.5]
        2) Apply controls → self.sim.step()
        3) Build minimal MuJoCo state for physics_sim
        4) Call physics_sim step → (None, physics_reward, physics_done, phys_info)
        5) Sync free‐particle positions back to MuJoCo
        6) Build next observation, return (obs, reward, done, info)
        """
        self.episode_step += 1

        scaled = np.array([
            action[0]*0.5,          # x: [-0.5, 0.5]
            action[1]*0.5,          # y: [-0.5, 0.5]
            action[2]*0.5 + 0.5,    # z: [0,1]
            action[3]*0.25 + 0.25   # plunger: [0,0.5]
        ], dtype=np.float32)

        # 2) Apply to MuJoCo
        self.data.ctrl[self.actuator_id['x']]       = scaled[0]
        self.data.ctrl[self.actuator_id['y']]       = scaled[1]
        self.data.ctrl[self.actuator_id['z']]       = scaled[2]
        self.data.ctrl[self.actuator_id['plunger']] = scaled[3]
        self.sim.step()

        # 3) Build minimal MuJoCo state for physics_sim
        mujoco_state = {
            'x_pos': float(self.data.qpos[self.joint_qposadr['x']]),
            'y_pos': float(self.data.qpos[self.joint_qposadr['y']]),
            'z_pos': float(self.data.qpos[self.joint_qposadr['z']])
        }

        # 4) Step physics_sim (pass normalized plunger action = action[3])
        _, physics_reward, physics_done, physics_info = self.env_wrapper.step(
            mujoco_state,
            np.array([scaled[0], scaled[1], scaled[2], action[3]], dtype=np.float32)
        )

        # 5) Sync free particles → MuJoCo
        self._sync_particles_to_mujoco()

        # 6) Build combined observation
        observation = self._get_observation()

        # Reward = physics_reward
        reward = float(physics_reward)

        # Done if any of: max steps, physics says done, or manual final check
        done = (
            (self.episode_step >= self.max_episode_steps)
            or physics_done
            or self._check_task_completion()
        )

        info = {
            'physics_info': physics_info,
            'episode_step': self.episode_step,
            'particles_held': len(self.physics_sim.held_particles),
            'pipette_state': physics_info['pipette_state'],
            'task_phase': physics_info['task_phase'],
            'current_well': self.physics_sim._get_current_well(),
            'task_completed': self._check_task_completion(),
            'reward_breakdown': physics_info,  # includes suction, held count, etc.
            'recent_events': {
                'aspirations': len([
                    e for e in self.physics_sim.aspiration_events
                    if (self.physics_sim.time - e.timestamp) < 1.0
                ]),
                'dispensing': len([
                    e for e in self.physics_sim.dispensing_events
                    if (self.physics_sim.time - e.timestamp) < 1.0
                ]),
                'ball_losses': len([
                    e for e in self.physics_sim.ball_loss_events
                    if (self.physics_sim.time - e.timestamp) < 1.0
                ]),
                'phase_violations': len([
                    e for e in self.physics_sim.phase_violation_events
                    if (self.physics_sim.time - e.timestamp) < 1.0
                ])
            }
        }

        return observation, reward, done, info

    def _check_task_completion(self) -> bool:
        """
        Task is complete if ≥2 particles originally from well 1
        now lie within 0.05m of well_3 center = (0.12, 0.0, 0.01).
        """
        well_3_pos = np.array([0.12, 0.0, 0.01], dtype=np.float32)
        count = 0
        for p in self.physics_sim.particles:
            if p.original_well == 1:
                if np.linalg.norm(p.position[:2] - well_3_pos[:2]) < 0.05:
                    count += 1
        return (count >= 2)

    def reset(self) -> np.ndarray:
        """Reset MuJoCo sim & physics sim, respawn particles, return initial observation."""
        self.episode_step = 0
        self.sim.reset()
        self.physics_sim.reset()
        self._initialize_particles()
        return self._get_observation()

    def render(self, mode='human'):
        """Render via MuJoCo viewer or return 'rgb_array' if requested."""
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)

        if mode == 'human':
            self.viewer.render()
        elif mode == 'rgb_array':
            return self.sim.render(width=640, height=480, camera_name='top_view')
        else:
            super().render(mode=mode)

    def close(self):
        """Close the MuJoCo viewer if it exists."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def create_actor_critic_compatible_env():
    return IntegratedPipetteEnv()
