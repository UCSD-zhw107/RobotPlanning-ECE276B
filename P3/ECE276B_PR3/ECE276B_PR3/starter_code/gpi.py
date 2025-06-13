from dataclasses import dataclass
import os
import numpy as np
from value_function import ValueFunction, GridValueFunction
import utils
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
from numba import jit, prange

@dataclass
class GpiConfig:
    T: int=100
    time_step: float=0.5
    traj: callable=utils.lissajous
    obstacles: np.ndarray= np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    #ex_space: np.ndarray=np.linspace(-3, 3, 25)
    #ey_space: np.ndarray=np.linspace(-3, 3, 25)
    # error space max range
    e_max: float = 3.0  
    # fine grid range
    e_fine_range: float = 1.0  
    # fine grid points
    e_fine_points: int = 15   
    # coarse grid points
    e_coarse_points: int = 5  
    eth_space: np.ndarray=np.linspace(-np.pi, np.pi, 72)
    v_space: np.ndarray=np.linspace(0.1, 1, 8)
    w_space: np.ndarray=np.linspace(-1, 1, 15)
    Q: np.ndarray=np.array([[1.0, 0.0], [0.0, 1.0]])
    q: float=1.0  # NOTE: I found that q=2.0 generally have better results for the deterministic case
    R: np.ndarray=np.array([[0.5, 0.0], [0.0, 0.5]])
    gamma: float=0.9
    num_evals: int=10
    collision_margin: float=0.3
    V: GridValueFunction=None
    output_dir: str='./gpi_output'
    collision_weight: float=1e3
    safety_margin: float=0.2
    '''# used by feature-based value function
    v_ex_space: np.ndarray
    v_ey_space: np.ndarray
    v_etheta_space: np.ndarray
    v_alpha: float
    v_beta_t: float
    v_beta_e: float
    v_lr: float
    v_batch_size: int  # batch size if GPU memory is not enough'''


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        self.T = config.T
        self.ex_space = self._create_adaptive_grid(
            config.e_max, config.e_fine_range, 
            config.e_fine_points, config.e_coarse_points
        )
        self.ey_space = self._create_adaptive_grid(
            config.e_max, config.e_fine_range, 
            config.e_fine_points, config.e_coarse_points
        )
        self.eth_space = config.eth_space
        self.V = GridValueFunction(config.T, self.ex_space, self.ey_space, self.eth_space)
        self.v_space = config.v_space
        self.w_space = config.w_space
        self.noise_mean = 0.0
        self.noise_sigma = utils.sigma
        self.time_step = config.time_step
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.actual_x_bounds: tuple = (-3, 3)
        self.actual_y_bounds: tuple = (-3, 3)
        self.actual_theta_bounds: tuple = (-np.pi, np.pi)
        self.ref_traj = self._generate_reference_trajectory()
        self.inflated_obstacles = np.copy(self.config.obstacles)
        self.inflated_obstacles[:, 2] += self.config.collision_margin
        self._build_valid_mask()
        if os.path.exists(self.config.output_dir + "/policy.npy"):
            print("Loading precomputed policy...")
            self.policy = np.load(self.config.output_dir + "/policy.npy")
        else:
            self.policy = self.compute_policy(100)


    def _create_adaptive_grid(self, max_range, fine_range, fine_points, coarse_points):
        """
        Create an adaptive grid
        """
        # fine grid
        fine_grid = np.linspace(-fine_range, fine_range, fine_points)
        
        if fine_range < max_range:
            # coarse grid
            n_coarse_per_side = coarse_points // 2
            left_coarse = np.linspace(-max_range, -fine_range, n_coarse_per_side, endpoint=False)
            right_coarse = np.linspace(fine_range, max_range, n_coarse_per_side, endpoint=False)[1:]
            grid = np.concatenate([left_coarse, fine_grid, right_coarse])
        else:
            grid = fine_grid
            
        return np.sort(np.unique(grid))
    

    def _find_closest_index(self, grid, value):
        """
        Find the closest index in a non-uniform grid.
        """
        idx = np.searchsorted(grid, value)
        
        if idx == 0:
            return 0
        elif idx >= len(grid):
            return len(grid) - 1
        else:
            if abs(grid[idx-1] - value) < abs(grid[idx] - value):
                return idx - 1
            else:
                return idx

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
    
        cur_error_state = cur_state - cur_ref_state
        cur_error_state[2] = np.arctan2(np.sin(cur_error_state[2]), np.cos(cur_error_state[2]))
        ex_idx, ey_idx, eth_idx = self.state_metric_to_index(cur_error_state)
        t_idx = t % self.T
        v = self.policy[t_idx, ex_idx, ey_idx, eth_idx, 0]
        w = self.policy[t_idx, ex_idx, ey_idx, eth_idx, 1]
        control = np.array([v, w])
        #control = self.control_index_to_metric(v_idx, w_idx)
        return control

    def _build_valid_mask(self):
        nx, ny, nth = len(self.ex_space), len(self.ey_space), len(self.eth_space)
        self.valid_mask = np.zeros((self.T, nx, ny, nth), dtype=bool)

        for t in range(self.T):
            ref_state = self.ref_traj[t]
            for ex_idx, ex in enumerate(self.ex_space):
                for ey_idx, ey in enumerate(self.ey_space):
                    for eth_idx, eth in enumerate(self.eth_space):
                        err = np.array([ex, ey, eth])
                        self.valid_mask[t, ex_idx, ey_idx, eth_idx] = \
                            self.is_valid_error_state(err, ref_state)

    def _generate_reference_trajectory(self) -> np.ndarray:
        """
        Generate the reference trajectory.
        Args:
            cur_iter (int): current iteration
        Returns:
            np.ndarray: reference trajectory    
        """
        ref_traj = []
        for i in range(self.T):
            ref_traj.append(self.config.traj(i))
        return np.array(ref_traj)
    


    def state_metric_to_index(self, metric_state: np.ndarray) -> tuple:
        """
        Convert the metric state to grid indices according to your descretization design.
        Args:
            metric_state (np.ndarray): metric state
        Returns:
            tuple: grid indices
        """
        #ex_idx = np.argmin(np.abs(self.ex_space - metric_state[0]))
        #ey_idx = np.argmin(np.abs(self.ey_space - metric_state[1]))
        ex_idx = self._find_closest_index(self.ex_space, metric_state[0])
        ey_idx = self._find_closest_index(self.ey_space, metric_state[1])
        etheta = np.arctan2(np.sin(metric_state[2]), np.cos(metric_state[2]))
        angle_diffs = np.abs(self.eth_space - etheta)
        angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
        eth_idx = np.argmin(angle_diffs)
        return ex_idx, ey_idx, eth_idx

    def state_index_to_metric(self, state_index: tuple) -> np.ndarray:
        """
        Convert the grid indices to metric state according to your descretization design.
        Args:
            state_index (tuple): grid indices
        Returns:
            np.ndarray: metric state
        """
        ex_idx, ey_idx, eth_idx = state_index
        ex = self.ex_space[ex_idx]
        ey = self.ey_space[ey_idx]
        etheta = self.eth_space[eth_idx]
        
        return np.array([ex, ey, etheta])


    def control_metric_to_index(self, control_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            control_metric: [2, N] array of controls in metric space
        Returns:
            [N, ] array of indices in the control space
            """
        v: np.ndarray = np.argmin(np.abs(self.v_space - control_metric[0]))
        w: np.ndarray = np.argmin(np.abs(self.w_space - control_metric[1]))
        return v, w

    def control_index_to_metric(self, v_idx: np.ndarray, w_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            v: [N, ] array of indices in the v space
            w: [N, ] array of indices in the w space
        Returns:
            [2, N] array of controls in metric space
        """
        return np.array([self.v_space[v_idx], self.w_space[w_idx]])
        
    def is_valid_error_state(self, error_state: np.ndarray, ref_state: np.ndarray) -> bool:
        """
        Check if an error state is valid (i.e., actual state would be in bounds)
        """
        # Compute what the actual state would be
        actual_state = error_state + ref_state
        
        # Check position bounds
        if (actual_state[0] < self.actual_x_bounds[0] or 
            actual_state[0] > self.actual_x_bounds[1] or
            actual_state[1] < self.actual_y_bounds[0] or 
            actual_state[1] > self.actual_y_bounds[1]):
            return False
        
        # Check obstacle bounds
        '''safety_tolerance = 1e-6
        for obs in self.inflated_obstacles:
            obs_center = obs[:2]
            inflated_radius = obs[2]
            dist_squared = (actual_state[0] - obs_center[0])**2 + (actual_state[1] - obs_center[1])**2
            
            if dist_squared <= (inflated_radius + safety_tolerance)**2:
                return False'''
        
        return True
    

    def _batch_state_metric_to_index(self, metric_states: np.ndarray) -> np.ndarray:
        """
        Convert a batch of metric states to grid indices.
        """
        batch_size = metric_states.shape[0]
        indices = np.zeros((batch_size, 3), dtype=int)
        
        ex_indices = np.searchsorted(self.ex_space, metric_states[:, 0])
        ey_indices = np.searchsorted(self.ey_space, metric_states[:, 1])
        
        # x index
        ex_indices = np.clip(ex_indices, 0, len(self.ex_space) - 1)
        ex_left_idx = np.maximum(ex_indices - 1, 0)
        ex_right_idx = ex_indices
        left_vals = self.ex_space[ex_left_idx]
        right_vals = self.ex_space[ex_right_idx]
        left_dists = np.abs(left_vals - metric_states[:, 0])
        right_dists = np.abs(right_vals - metric_states[:, 0])
        use_left = (left_dists < right_dists) & (ex_indices > 0)
        indices[:, 0] = np.where(use_left, ex_left_idx, ex_right_idx)

        # y index
        ey_indices = np.clip(ey_indices, 0, len(self.ey_space) - 1)
        ey_left_idx = np.maximum(ey_indices - 1, 0)
        ey_right_idx = ey_indices
        left_vals = self.ey_space[ey_left_idx]
        right_vals = self.ey_space[ey_right_idx]
        left_dists = np.abs(left_vals - metric_states[:, 1])
        right_dists = np.abs(right_vals - metric_states[:, 1])
        use_left = (left_dists < right_dists) & (ey_indices > 0)
        indices[:, 1] = np.where(use_left, ey_left_idx, ey_right_idx)
        
        # angle indx
        etheta_normalized = np.arctan2(np.sin(metric_states[:, 2]), np.cos(metric_states[:, 2]))
        angle_diffs = np.abs(self.eth_space[np.newaxis, :] - etheta_normalized[:, np.newaxis])
        angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
        indices[:, 2] = np.argmin(angle_diffs, axis=1)
        
        return indices
    
    def _compute_next_error_states(self, error_states, controls, cur_ref_state, next_ref_state, ref_diff):
        """
        Args:
            error_states: [batch_size, 3] 
            controls: [batch_size, 2] 
            cur_ref_state, next_ref_state: [3] 
            ref_diff: [3] 
        Returns:
            [batch_size, 3]
        """
        
        theta_actual = error_states[:, 2] + cur_ref_state[2]  # [batch_size]
        cos_theta = np.cos(theta_actual)
        sin_theta = np.sin(theta_actual)
        f_batch = np.column_stack([
            controls[:, 0] * cos_theta,  
            controls[:, 0] * sin_theta,  
            controls[:, 1]               
        ])  # [batch_size, 3]
        
        
        next_states = error_states + self.time_step * f_batch + ref_diff
        next_states[:, 2] = np.arctan2(np.sin(next_states[:, 2]), np.cos(next_states[:, 2]))
        
        return next_states
    
        

    def compute_transition_matrix(self):
        
        print("Computing transition matrix ...")
        
        nx, ny, nth = len(self.ex_space), len(self.ey_space), len(self.eth_space)
        nv, nw = len(self.v_space), len(self.w_space)
        
        transitions = np.zeros((self.T, nx, ny, nth, nv, nw, 3), dtype=int)
        
        for t in range(self.T):
            print(f"Processing time step {t}/{self.T}")
            
        
            cur_ref_state = self.ref_traj[t]
            next_ref_state = self.ref_traj[(t + 1) % self.T]
            ref_diff = np.zeros(3)
            ref_diff[0] = cur_ref_state[0] - next_ref_state[0]
            ref_diff[1] = cur_ref_state[1] - next_ref_state[1]
            angle_diff = cur_ref_state[2] - next_ref_state[2]
            ref_diff[2] = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            
            ex_indices, ey_indices, eth_indices, v_indices, w_indices = np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(nth), 
                np.arange(nv), np.arange(nw), indexing='ij'
            )
            
        
            batch_size = nx * ny * nth * nv * nw
            ex_flat = ex_indices.flatten()
            ey_flat = ey_indices.flatten()
            eth_flat = eth_indices.flatten()
            v_flat = v_indices.flatten()
            w_flat = w_indices.flatten()
            
            
            error_states = np.column_stack([
                self.ex_space[ex_flat],
                self.ey_space[ey_flat],
                self.eth_space[eth_flat]
            ])  # [batch_size, 3]
            
            controls = np.column_stack([
                self.v_space[v_flat],
                self.w_space[w_flat]
            ])  # [batch_size, 2]
            
            
            next_states = self._compute_next_error_states(
                error_states, controls, cur_ref_state, next_ref_state, ref_diff
            )
            
            
            next_indices = self._batch_state_metric_to_index(next_states)
            
            
            next_indices_reshaped = next_indices.reshape(nx, ny, nth, nv, nw, 3)
            transitions[t] = next_indices_reshaped
        
        print("Computation complete!")
        np.save(self.config.output_dir + "/transitions.npy", transitions)
        return transitions
    

    

    def _compute_stage_costs(self, cur_error_states: np.ndarray, controls: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Args:
            cur_error_states: [batch_size, 3] 
            controls: [batch_size, 2] 
            cur_ref_state: [3] 
        Returns:
            [batch_size] 
        """
        batch_size = cur_error_states.shape[0]
        
        
        pos_errors = cur_error_states[:, :2]  # [batch_size, 2]
        
        Q_expanded = np.broadcast_to(self.config.Q, (batch_size, 2, 2))
        pos_costs = np.einsum('bi,bij,bj->b', pos_errors, Q_expanded, pos_errors)
        
       
        angle_errors = cur_error_states[:, 2]  # [batch_size]
        angle_costs = self.config.q * (1 - np.cos(angle_errors))**2
        
        R_expanded = np.broadcast_to(self.config.R, (batch_size, 2, 2))
        control_costs = np.einsum('bi,bij,bj->b', controls, R_expanded, controls)

                     
        actual_state = cur_error_states + cur_ref_state
        actual_state[:, 2] = np.arctan2(np.sin(actual_state[:, 2]), np.cos(actual_state[:, 2]))

        obs_costs = np.zeros(batch_size, dtype=float)
        collision_mask = np.zeros(batch_size, dtype=bool)  
        for obs in self.inflated_obstacles:
            obs_center = obs[:2]
            inflated_radius = obs[2]
            dist = np.linalg.norm(actual_state[:, :2] - obs_center, axis=1) - inflated_radius
            collision_mask |= (dist < 0)
            within = (dist >= 0) & (dist < self.config.safety_margin)
            obs_costs[within] += self.config.collision_weight * (self.config.safety_margin - dist[within])

        total_costs = pos_costs + angle_costs + control_costs + obs_costs
        total_costs[collision_mask] = np.inf
        return total_costs
    
    def compute_stage_costs(self):
    
        print("Computing stage costs...")
        
        nx, ny, nth = len(self.ex_space), len(self.ey_space), len(self.eth_space)
        nv, nw = len(self.v_space), len(self.w_space)
        
        stage_costs = np.zeros((self.T, nx, ny, nth, nv, nw))
        
        for t in range(self.T):
            print(f"Processing time step {t}/{self.T}")
            
            cur_ref_state = np.array(self.config.traj(t))
            
            
            ex_indices, ey_indices, eth_indices, v_indices, w_indices = np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(nth),
                np.arange(nv), np.arange(nw), indexing='ij'
            )
            
            
            batch_size = nx * ny * nth * nv * nw
            ex_flat = ex_indices.flatten()
            ey_flat = ey_indices.flatten()
            eth_flat = eth_indices.flatten()
            v_flat = v_indices.flatten()
            w_flat = w_indices.flatten()
            
            
            error_states = np.column_stack([
                self.ex_space[ex_flat],
                self.ey_space[ey_flat],
                self.eth_space[eth_flat]
            ])
            
            controls = np.column_stack([
                self.v_space[v_flat],
                self.w_space[w_flat]
            ])
            
            
            costs = self._compute_stage_costs(error_states, controls, cur_ref_state)
            
            
            valid_mask_flat = self.valid_mask[t].flatten()
            valid_indices = np.repeat(valid_mask_flat, nv * nw)
            costs[~valid_indices] = np.inf
            
            stage_costs[t] = costs.reshape(nx, ny, nth, nv, nw)
        
        print("Computation complete!")
        np.save(self.config.output_dir + "/stage_costs.npy", stage_costs)
        return stage_costs


    def init_value_function(self):
        """
        Initialize the value function.
        """
        self.V.value = np.zeros((self.T, len(self.ex_space), len(self.ey_space), len(self.eth_space)))
    

    def init_policy(self):
        """
        Initialize the policy storage.
        """
        # Store optimal action indices for each state
        self.policy = np.zeros((self.T, len(self.ex_space), len(self.ey_space), 
                               len(self.eth_space), 2), dtype=float)

    @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        nx = len(self.ex_space)
        ny = len(self.ey_space)
        nth = len(self.eth_space)
        nv = len(self.v_space)
        nw = len(self.w_space)
        policy_changed = False
        for t in range(self.T):
            next_t = (t + 1) % self.T
            new_policy, changed = self._policy_improvement_numba(
                self.stage_costs[t], 
                self.transitions[t], 
                self.V.value[next_t],
                self.policy[t],
                self.config.gamma,
                self.v_space,
                self.w_space,
                nx, ny, nth, nv, nw
            )
            if changed:
                policy_changed = True
                self.policy[t] = new_policy
        return policy_changed
    

    @staticmethod
    @jit(nopython=True)
    def _policy_improvement_numba(
        stage_costs, transitions, next_V, current_policy, gamma, v_space, w_space, nx, ny, nth, nv, nw
    ):
        """
        Policy improvement step of the GPI algorithm.
        """
        new_policy = current_policy.copy()
        policy_changed = False
        for ex_idx in prange(nx):
            for ey_idx in range(ny):
                for eth_idx in range(nth):
                    best_value = np.inf
                    best_v = current_policy[ex_idx, ey_idx, eth_idx, 0]
                    best_w = current_policy[ex_idx, ey_idx, eth_idx, 1]
                    for v_idx in range(nv):
                        for w_idx in range(nw):
                            cost = stage_costs[ex_idx, ey_idx, eth_idx, v_idx, w_idx]
                            if np.isinf(cost):                
                                continue
                            next_state = transitions[ex_idx, ey_idx, eth_idx, v_idx, w_idx]
                            nex_idx, ney_idx, neth_idx = next_state[0], next_state[1], next_state[2]
                            expected_value = next_V[nex_idx, ney_idx, neth_idx]
                            value = cost + gamma * expected_value
                            if value < best_value:
                                best_value = value
                                best_v = v_space[v_idx]
                                best_w = w_space[w_idx]
                    if (current_policy[ex_idx, ey_idx, eth_idx, 0] != best_v or 
                        current_policy[ex_idx, ey_idx, eth_idx, 1] != best_w):
                        policy_changed = True
                    
                    new_policy[ex_idx, ey_idx, eth_idx, 0] = best_v
                    new_policy[ex_idx, ey_idx, eth_idx, 1] = best_w
        return new_policy, policy_changed
                                    
    @utils.timer
    def policy_evaluation(self, num_evals=1):
        """
        Policy evaluation step of the GPI algorithm.
        """
        nx = len(self.ex_space)
        ny = len(self.ey_space)
        nth = len(self.eth_space)

        for _ in range(num_evals):
            for t in range(self.T):
                next_t = (t + 1) % self.T
                
                self.V.value[t] = self._policy_evaluation_numba(
                    self.stage_costs[t],
                    self.transitions[t],
                    self.V.value[next_t],
                    self.policy[t],
                    self.config.gamma,
                    self.v_space,
                    self.w_space,
                    nx, ny, nth
                )
    
    @staticmethod
    @jit(nopython=True)
    def _policy_evaluation_numba(
        stage_costs, transitions, next_V, policy, gamma, v_space, w_space, nx, ny, nth
    ):
        """
        Policy evaluation step of the GPI algorithm.
        """
        V_new = np.full((nx,ny,nth), np.inf)
        
        for ex_idx in prange(nx):
            for ey_idx in range(ny):
                for eth_idx in range(nth):
                    v = policy[ex_idx, ey_idx, eth_idx, 0]
                    w = policy[ex_idx, ey_idx, eth_idx, 1]

                    v_idx = np.argmin(np.abs(v_space - v))
                    w_idx = np.argmin(np.abs(w_space - w))
                    
                    cost = stage_costs[ex_idx, ey_idx, eth_idx, v_idx, w_idx]
                    if np.isinf(cost):
                        continue
                    next_state = transitions[ex_idx, ey_idx, eth_idx, v_idx, w_idx]
                    nex_idx, ney_idx, neth_idx = next_state[0], next_state[1], next_state[2]
                    expected_value = next_V[nex_idx, ney_idx, neth_idx]
                    V_new[ex_idx, ey_idx, eth_idx] = cost + gamma * expected_value
        return V_new

            

    def compute_policy(self, num_iters: int) -> None:
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        # Load or compute transitions and costs
        if os.path.exists(self.config.output_dir + "/transitions.npy"):
            print("Loading precomputed transitions...")
            self.transitions = np.load(self.config.output_dir + "/transitions.npy")
        else:
            self.transitions = self.compute_transition_matrix()
        
        if os.path.exists(self.config.output_dir + "/stage_costs.npy"):
            print("Loading precomputed stage costs...")
            self.stage_costs = np.load(self.config.output_dir + "/stage_costs.npy")
        else:
            self.stage_costs = self.compute_stage_costs()

        # Initialize value function and policy
        self.init_value_function()
        self.init_policy()
        print("Starting GPI iterations...")
        for iteration in range(num_iters):
            print(f"\nIteration {iteration + 1}/{num_iters}")

            # Policy evaluation
            self.policy_evaluation(num_evals=self.config.num_evals)
            
            # Policy improvement
            policy_changed = self.policy_improvement()
            
            # Check convergence
            if not policy_changed and iteration > 0:
                print(f"Policy converged after {iteration + 1} iterations!")
                break
            
            
            # Save intermediate results
            #if (iteration + 1) % 10 == 0:
                #self.save_policy(iteration + 1)
        
        # Save final policy
        self.save_policy(num_iters)
        print("GPI complete!")
        return self.policy

    def save_policy(self, iteration):
        """Save policy and value function to disk."""
        np.save(f"{self.config.output_dir}/policy.npy", self.policy)
        np.save(f"{self.config.output_dir}/value.npy", self.V.value)