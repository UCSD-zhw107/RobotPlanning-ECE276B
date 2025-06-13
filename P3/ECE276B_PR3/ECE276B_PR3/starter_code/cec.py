import casadi
import numpy as np
import utils


class CEC:
    def __init__(self, time_step: float, sim_time: float, obstacles: np.ndarray = None) -> None:
        self.time_step = time_step
        self.sim_time = sim_time
        self.horizon = int(sim_time // time_step)
        self.q = 1.0 # stage cost of deviation from ref orientation
        self.p = np.array([[1.0, 0.0], [0.0, 1.0]])  # stage cost of deviation from ref position
        self.r = np.array([[0.1, 0.0], [0.0, 0.1]])  # stage cost for using control effort 
        self.terminal_cost = 10.0 # terminal cost of deviation from ref position
        self.traj = utils.lissajous
        self.boundaries = [-3, 3]
        self.angle_boundaries = [-np.pi, np.pi]
        self.gamma = 0.9
        self.plan_horizon = 10
        self.u_min = [-1.0, -1.0]  # [v_min, omega_min]
        self.u_max = [1.0, 1.0]    # [v_max, omega_max]

        self.robot_radius = 0.3
        self.obstacles = obstacles if obstacles is not None else np.array([])
        if len(self.obstacles) > 0:
            self.inflated_obstacles = np.copy(self.obstacles)
            self.inflated_obstacles[:, 2] += self.robot_radius # inflated by robot radius
        
    
    def _generate_reference_trajectory(self, cur_iter: int, plan_time: int) -> np.ndarray:
        """
        Generate the reference trajectory.
        Args:
            cur_iter (int): current iteration
        Returns:
            np.ndarray: reference trajectory
        """
        ref_traj = []
        for i in range(plan_time + 1):
            ref_traj.append(self.traj(cur_iter + i))
        return np.array(ref_traj)
    

    def _car_dynamics(self, cur_error_state, control, cur_ref_state, next_ref_state) -> np.ndarray:
        """
        Car dynamics.
        Args:
            cur_state (np.ndarray): current state
            control (np.ndarray): control input
        """
        # control term
        theta = cur_error_state[2] + cur_ref_state[2]
        rot_3d_z = casadi.vertcat(
            casadi.horzcat(casadi.cos(theta), 0),
            casadi.horzcat(casadi.sin(theta), 0),
            casadi.horzcat(0, 1)
        )
        f = rot_3d_z @ control

        # error term
        e = cur_error_state

        # reference term
        ref_diff = cur_ref_state - next_ref_state

        e_next = e + self.time_step * f + ref_diff
        e_next[2] = np.arctan2(np.sin(e_next[2]), np.cos(e_next[2]))
        return e_next
    




    def __call__(self, cur_iter: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            cur_iter (int): current iteration
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # generate reference trajectory
        t = min(self.horizon - cur_iter, self.plan_horizon)
        if t <= 0:
            return np.array([0.0, 0.0])
        ref_traj = self._generate_reference_trajectory(cur_iter, t)

        # define optimization variables
        x = casadi.SX.sym("x", 3, t + 1)
        u = casadi.SX.sym("u", 2, t)

        # define optimization constraints
        g = []
        lbg = []
        ubg = []
        
        # initial state constraint
        cur_error_state = cur_state - cur_ref_state
        cur_error_state[2] = np.arctan2(np.sin(cur_error_state[2]), np.cos(cur_error_state[2]))
        g.append(x[:, 0] - cur_error_state)
        lbg.extend([0, 0, 0])
        ubg.extend([0, 0, 0])

        # dynamic constraint
        for i in range(t):
            next_error_state = self._car_dynamics(x[:, i], u[:, i], ref_traj[i], ref_traj[i+1])
            g.append(x[:, i + 1] - next_error_state)
            lbg.extend([0, 0, 0])
            ubg.extend([0, 0, 0])

        # Add state constraints (actual car state bounds)
        for i in range(t + 1):
            # actual state: x = e + r
            ref_state = ref_traj[i]
            actual_state = x[:, i] + ref_state
            
            # Position bounds
            g.append(actual_state[0])  # x position
            lbg.append(self.boundaries[0])
            ubg.append(self.boundaries[1])
            
            g.append(actual_state[1])  # y position
            lbg.append(self.boundaries[0])
            ubg.append(self.boundaries[1])
            
            # Angle bounds
            g.append(casadi.sin(actual_state[2]))
            lbg.append(-1.0)  # sin(theta) in [-1, 1]
            ubg.append(1.0)
            
            g.append(casadi.cos(actual_state[2]))
            lbg.append(-1.0)  # cos(theta) in [-1, 1]
            ubg.append(1.0)

            # obstacle avoidance
            if len(self.obstacles) > 0:
                for obs in self.inflated_obstacles:
                    obs_center = obs[:2]
                    inflated_radius = obs[2]
                    dist_squared = (actual_state[0] - obs_center[0])**2 + (actual_state[1] - obs_center[1])**2
                    g.append(dist_squared)
                    lbg.append(inflated_radius**2)
                    ubg.append(np.inf)

        # Objective function
        f = 0
        for i in range(t):
            # Position error cost
            pos_error = x[:2, i]
            pos_cost = casadi.dot(pos_error, self.p @ pos_error)
            
            # Orientation error cost
            angle_cost = self.q * (1 - casadi.cos(x[2, i]))**2
            
            # Control effort cost
            control_cost = casadi.dot(u[:, i], self.r @ u[:, i])
            f += self.gamma**i * (pos_cost + angle_cost + control_cost)
        terminal_pos_error = x[:2, t]
        f += self.terminal_cost * casadi.dot(terminal_pos_error, terminal_pos_error)


        # Bounds
        lbw = []
        ubw = []
        # error state bounds
        for i in range(t + 1):
            lbw.extend([-np.inf, -np.inf, -np.pi])  # might use -2*np.pi
            ubw.extend([np.inf, np.inf, np.pi])
        # control bounds
        for i in range(t):
            lbw.extend([self.u_min[0], self.u_min[1]])
            ubw.extend([self.u_max[0], self.u_max[1]])
        # initial guess
        w = casadi.vertcat(casadi.reshape(x, -1, 1), casadi.reshape(u, -1, 1))
        w0 = np.zeros(w.shape[0])
        w0[:3] = cur_error_state
        
        nlp = {
            "x": w,
            "f": f,
            "g": casadi.vertcat(*g)
        }
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
        }
        solver = casadi.nlpsol("S", "ipopt", nlp, opts)
        sol = solver(
            x0=w0,
            lbx=lbw,
            ubx=ubw,
            lbg=lbg,
            ubg=ubg
        )
        w_opt = sol["x"].full().flatten()

        control_start_idx = 3 * (t + 1)
        u_opt = w_opt[control_start_idx:control_start_idx + 2 * t]
        u_opt = u_opt.reshape((t, 2))
        return u_opt[0, :]
