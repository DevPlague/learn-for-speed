# Physics guide

How the car, the track and the sensors work, written so that every teammate can explain them in the oral defense. Each formula is matched to the function that implements it, and each number in this guide has a test in `tests/test_sim_physics.py`.

The car is a **bicycle model extended with sideways slip**. It is not a full vehicle dynamics model. There is no mass, no tire curve and no load transfer. The goal is a car that is simple to explain, fast to simulate, and forces the agent to brake for corners.

## Conventions
- Units are meters, seconds and radians. Speed is in m/s and acceleration in m/s^2.
- The world frame has x to the right and y up. Yaw $\psi$ is measured counterclockwise from the x axis.
- The car frame has x forward and y to the left. A positive steering angle turns left.
- The car position $(x, y)$ is the **rear axle**. Lidar, road checks and barrier checks all use this one point.

## State, action and parameters
State, `[x, y, yaw, v, steer, v_lat]`, read by baselines as `env.state`.

| Symbol | Code | Unit | Meaning |
|---|---|---|---|
| $x, y$ | `x`, `y` | m | Rear axle position in the world |
| $\psi$ | `yaw` | rad | Direction the nose points |
| $v_x$ | `v` | m/s | Forward speed in the car frame, negative in reverse |
| $\delta$ | `steer` | rad | Angle of the front wheels |
| $v_y$ | `v_lat` | m/s | Sideways slip speed in the car frame, positive to the left |

Action, `[steer, throttle]`, both clipped to [-1, 1].

| Symbol | Meaning |
|---|---|
| $u_s$ | Steering command, 1 is full left |
| $u_t$ | Throttle command. Pushing against the motion brakes, in either direction. Otherwise positive accelerates forward and negative drives backwards |

Parameters, all in `rust/src/car.rs`. Changing one needs a rebuild, which `make play` and `make test` do by themselves.

| Symbol | Code | Value | Meaning |
|---|---|---|---|
| $\Delta t$ | `DT` | 0.05 s | Time step, 20 steps per second |
| $L$ | `WHEELBASE` | 2.5 m | Distance between the axles |
| $\delta_{max}$ | `MAX_STEER` | 0.5 rad | Largest wheel angle, about 28.6 degrees |
| $\rho$ | `STEER_RATE` | 2 rad/s | How fast the wheels turn, full lock in 0.25 s |
| $V_{max}$ | `MAX_SPEED` | 20 m/s | Forward speed cap, 72 km/h |
| $V_{rev}$ | `REVERSE_SPEED` | 5 m/s | Reverse speed cap, 18 km/h |
| $A$ | `ACCEL` | 6 m/s^2 | Acceleration at full throttle |
| $B$ | `BRAKE` | 10 m/s^2 | Deceleration at full brake |
| $D_0$ | `DRAG` | 0.05 1/s | Linear drag |
| $G_{road}$ | `GRIP` | 8 m/s^2 | Sideways acceleration the tires hold on the road |
| $Y_{road}$ | `YAW_GRIP` | 14 m/s^2 | How hard the nose may turn on the road, past $G$ the car slides |
| $D_g$ | `GRASS_DRAG` | 0.8 1/s | Extra drag on grass |
| $G_{grass}$ | `GRASS_GRIP` | 4 m/s^2 | Tire grip on grass |

## One step, in order
The order is part of the model. `step_car` in `rust/src/car.rs` does these six things. Integration is first order. The position is semi-implicit, it uses the yaw and speeds just computed. Drag uses the old speed and stage 4 uses the old values of both axes, which is explicit Euler.

### 1. Steering actuator
The wheels move toward the command at a limited rate, so the agent cannot flick them instantly.

$$\delta' = \delta + \mathrm{clip}(u_s \delta_{max} - \delta,\ -\rho\Delta t,\ \rho\Delta t)$$

### 2. Forward speed
Pushing against the motion brakes, otherwise the motor acts. Braking is $v_x u_t < 0$.

$$v^* = v_x + (F u_t - D v_x)\Delta t \qquad F = \begin{cases} B & \text{braking} \\ A & \text{otherwise} \end{cases} \qquad D = \begin{cases} D_0 + D_g & \text{on grass} \\ D_0 & \text{on road} \end{cases}$$

If braking would change the sign of the speed, $v^* = 0$ instead, so the brakes stop the car and reverse only starts on the next step. Then $v^*$ is clipped to $[-V_{rev}, V_{max}]$.

Results of this equation.
- Top forward speed on the road is the cap of 20 m/s, since the balance $A / D_0 = 120$ m/s is far above it. The cap acts on $v_x$ only. Slip adds to it, so the total speed can reach about 22 m/s while sliding, and the screen can read about 79 km/h.
- The speed that full throttle holds on grass is $A / (D_0 + D_g) = 7.06$ m/s, about 25 km/h. This is a balance, not a limit. A car entering the grass faster slows down gradually, and can reach the barrier at 13 m/s or more.
- From 0 to 20 m/s takes $-\frac{1}{D_0}\ln(1 - \frac{20 D_0}{A}) = 3.65$ s.
- From 20 m/s, full brake stops in $\frac{1}{D_0}\ln(1 + \frac{20 D_0}{B}) = 1.91$ s. The exact distance is 18.8 m, and 20 m without drag from $v^2 / 2B$. The simulation travels 18.2 m, half a step less, because the position uses the speed already updated in the same step.
- Coasting decays as $v_x (1 - D_0\Delta t)^n$ after $n$ steps. It never reaches zero by itself, the no progress rule below ends such episodes.

### 3. Yaw rate
This is the bicycle model. Two wheels at distance $L$, the rear one rolling straight and the front one at angle $\delta$, turn around a point at radius $R = L / \tan\delta$. So the yaw rate is $r = v / R$.

The turn this asks for needs a sideways acceleration $a_{dem} = v r = v^2 \tan\delta / L$. The nose may only turn as hard as $Y$, with $Y = Y_{road}$ on the road and $Y = Y_{road} G_{grass} / G_{road} = 7$ m/s^2 on grass. So, when $|v^*| > 10^{-3}$,

$$\tan\delta_{eff} = \mathrm{clip}\left(\tan\delta',\ -\frac{Y L}{v^{*2}},\ \frac{Y L}{v^{*2}}\right) \qquad \Delta\psi = \frac{v^* \tan\delta_{eff}}{L}\Delta t$$

The tightest possible turn has radius $L / \tan\delta_{max} = 4.58$ m. In reverse $v^*$ is negative, so the same wheel angle turns the car the other way, like a real car.

### 4. Slip from turning
These are the textbook equations of motion in a rotating frame, $\dot{v}_x = a_x + r v_y$ and $\dot{v}_y = a_y - r v_x$. The car rotates while its velocity keeps its direction, so forward speed appears as sideways speed.

$$v_x' = \mathrm{clip}(v^* + \Delta\psi\, v_y,\ -V_{rev},\ V_{max}) \qquad v_y^* = v_y - \Delta\psi\, v^*$$

In a left turn $\Delta\psi > 0$, so $v_y^*$ goes negative, which means the car slides to the outside of the turn, as expected.

### 5. Tire friction
The tires oppose the sideways motion with at most the grip $G$. This is friction with a limit, so a small slip is fully cancelled and a large one only shrinks.

$$v_y' = v_y^* - \mathrm{clip}(v_y^*,\ -G\Delta t,\ G\Delta t)$$

### 6. Position
The new yaw and speeds move the car, with the slip included.

$$\psi' = \psi + \Delta\psi$$
$$x' = x + (v_x' \cos\psi' - v_y' \sin\psi')\Delta t$$
$$y' = y + (v_x' \sin\psi' + v_y' \cos\psi')\Delta t$$

## Grip and drift
Stages 4 and 5 together give the drift. In one step the turn creates a slip of $|\Delta\psi\, v^*| = a\,\Delta t$, with $a = \min(a_{dem}, Y)$, and the tires can remove $G\Delta t$.

- **Grip.** If $a \le G$ the slip is removed in the same step, so $v_y$ stays at zero and the car follows the bicycle path. A corner of radius $R$ can be taken without sliding up to $v = \sqrt{G R}$. That is 11 m/s for 15 m and 15.5 m/s for 30 m.
- **Drift.** If $a > G$ the rest accumulates. The slip grows at $a - G$, which is at most $Y - G = 6$ m/s^2 on the road and $7 - 4 = 3$ m/s^2 on grass.
- **Recovery.** With straight wheels the slip shrinks at $G$, so a slip of 6 m/s is gone in 0.75 s on the road.
- **Spin.** Slip keeps growing only while the car is fast enough. From 20 m/s at full lock and no throttle, the slip peaks at 11.7 m/s after about 2 s and then returns to zero as the car slows, with the nose turned about 240 degrees. With full throttle held, the slip peaks at 18.9 m/s after about 3 s, then the car settles into a steady circle at $v_x = 5.8$ and $v_y = -4.3$ m/s, spinning for as long as the throttle is held.

With the wheels at full lock, grip ends at $\sqrt{G L / \tan\delta_{max}} = 6.05$ m/s on the road and 4.28 m/s on grass. The nose reaches its limit $Y$ at 8 m/s on the road.

## Grass
The car is on grass when $|lateral| > w/2$, see the track section. This changes three things in the step above, the drag $D$, the grip $G$ and the yaw limit $Y$. The flag comes from the end of the previous step, so grass takes effect one step late, about 1.1 m of travel at most.

## Track geometry
Built in `sim/track.py` with NumPy and scipy, once per reset. The same seed and difficulty $d$ always give the same track.

1. **Control points.** $k = \mathrm{round}(6 + 6d)$ points around a circle of radius $R_0 = 70$ m, with $\epsilon_i \sim U(-0.3, 0.3)$, $\xi_i \sim U(-1, 1)$ and noise $a = 0.1 + 0.4d$. Difficulty is continuous, $k$ takes the 7 values 6 to 12 and $a$ goes from 0.1 to 0.5.
$$\theta_i = \frac{2\pi(i + \epsilon_i)}{k} \qquad r_i = R_0(1 + a\,\xi_i)$$
2. **Spline.** A periodic cubic spline through the points, so position, tangent and curvature are all continuous around the loop.
3. **Even spacing.** Resampled by arc length, about one point per meter.
4. **Heading.** $h_i = \mathrm{atan2}(p_{i+1} - p_{i-1})$ by central difference, with the tangent $t_i = (\cos h_i, \sin h_i)$ and the normal $n_i = (-\sin h_i, \cos h_i)$.
5. **Edges and barriers.** The road width is $w = 8$ m and the grass gap is $g = 8$ m. Edges are at $p_i \pm n_i w/2$ and barriers at $p_i \pm n_i (w/2 + g)$, which is 12 m from the centerline.
6. **Easing tight corners.** A curve offset by a distance larger than its radius folds on itself, which would break the inner barrier, so no corner may be tighter than 12 m. Where the turn between two segments $\tau_i$ is larger than $|p_{i+1} - p_i| / 14$, the excess moves half to each neighbour and this repeats until no excess is left. Every corner keeps its total angle and only its radius grows, and the small gap this opens at the start line is spread linearly over the whole loop. Before this step the generator redrew any tight track with 10% less noise, so 98% of the tracks at difficulty 1 ended up smoother than at 0.5.
7. **Validity.** A candidate is rejected and redrawn, with 10% less noise, when either rule fails.
   - Tightest radius, from $\kappa_i = |h_{i+1} - h_i| / |p_{i+1} - p_i|$, must satisfy $1/\kappa \ge 12$ m. Easing fixes about 95% of the tightest splines, the rest are redrawn.
   - Points over 32 m apart along the road must stay over 17 m apart in space. Since $12 + 4 + 1 = 17$, no barrier stands on another part of the road. Barriers of two road parts closer than 24 m would cross each other, which this rule allows.

After 50 rejected tries `generate_track` raises an error. This has not happened in testing.

Measured on validation seeds 1000 to 1049, the difficulty now changes the track as intended.

| Difficulty | Control points | Points tighter than 25 m | Total turning, in laps | Generation |
|---|---|---|---|---|
| 0 | 6 | 2% | 1.00 | 8 ms |
| 0.5 | 9 | 10% | 1.29 | 13 ms |
| 1 | 12 | 26% | 2.44 | 35 ms |

Pure pursuit laps all 50 tracks at every difficulty, in 29.5 s, 32.3 s and 41.5 s.

## Position along the track
`locate` and `frenet` in `rust/src/track.rs`.

The nearest centerline point $i$ is searched within 20 points of the previous one. The car moves about 1.1 m per step at most. The index can move faster than the car near the inner barrier of a tight corner, so this is a checked result and not a guarantee, and `test_search_window_finds_the_nearest_point` confirms the search finds the true nearest point during hard driving. From that point,

$$\Delta = (x, y) - p_i \qquad along = \Delta \cdot t_i \qquad lateral = \Delta \cdot n_i \qquad frac = \frac{(s_i + along) \bmod \ell}{\ell}$$

where $s_i$ is the arc length of the point and $\ell$ the lap length. A positive lateral offset is to the left. The env adds the change in $frac$ to the lap progress, after unwrapping it across the start line. Driving backwards from the start makes the progress negative, and `info["progress"]` clips it to [0, 1].

$$\Delta p = ((frac - frac_{prev} + 0.5) \bmod 1) - 0.5$$

## Lidar
`angles` and `cast` in `rust/src/lidar.rs`.

There are 25 rays with a range of $R_l = 30$ m. The angles relative to the nose are denser in front, with one ray straight ahead.

$$u_k = \frac{k}{12.5} \quad k = -12 \ldots 12 \qquad \alpha_k = \pi\, \mathrm{sign}(u_k)\, |u_k|^{1.5}$$

That gives 4.1 degrees between the first rays and reaches 169.3 degrees, leaving a gap of 21.4 degrees straight behind.

Each ray has origin $o = (x, y)$ and direction $d = (\cos(\psi + \alpha_k), \sin(\psi + \alpha_k))$. Each segment starts at $a$ with direction $e$. Solving $o + t d = a + u e$ by taking the 2D cross product with $e$ and then with $d$ gives

$$t = \frac{(a - o) \times e}{d \times e} \qquad u = \frac{(a - o) \times d}{d \times e}$$

The ray hits when $t > 0$ and $0 \le u \le 1$, at distance $t$. The reading is the smallest hit, capped at $R_l$. A parallel ray divides by zero, and the resulting NaN or infinity fails the test, so it counts as no hit. Any segment whose start is farther than $R_l$ plus its own length cannot be reached, by the triangle inequality, so it is skipped. That skip does not change the result.

Rays stop at road edges and at barriers, so the painted road edge counts as visible to the sensor. On the road the agent normally sees only the edges, and from the grass it can see the barrier too. The reading does not say which of the two was hit.

## Observation
`_obs` in `sim/env.py`. Sensors only, 28 values, clipped to [-1, 1]. The track map is never observed.

| Values | Scaling |
|---|---|
| 25 lidar readings | $2 h / R_l - 1$ |
| Forward speed | $v_x / V_{max}$ |
| Steering angle | $\delta / \delta_{max}$ |
| Sideways slip | $v_y / V_{max}$ |

The slip uses the same scale as the speed, because a spin under full throttle reaches about 19 m/s of slip and a smaller scale would clip it.

## When an attempt ends
`step` in `sim/env.py`. All checks use the rear axle point, after the move.

| Rule | Condition | Result |
|---|---|---|
| Lap done | Progress reaches 1 | Terminated, `lap_done` |
| Barrier | $|lateral| > 12$ m | Terminated, `off_road` |
| Too long on grass | Over 10 s in a row | Terminated, `off_road` |
| No progress | Under 2 m gained in 10 s | Terminated, `stuck` |
| Timeout | 120 s | Truncated |

The next episode starts on the start line, at the first centerline point, stopped and pointing along the road.

## Formula to code
| Section | Code |
|---|---|
| Steering, forward speed, yaw, slip, friction, position | `step_car` in `rust/src/car.rs` |
| Ray angles and intersection | `angles`, `cast` in `rust/src/lidar.rs` |
| Nearest point, progress, lateral offset | `locate`, `frenet` in `rust/src/track.rs` |
| One call per env step | `TrackCore.step` in `rust/src/lib.rs` |
| Point count, noise and redraw loop | `generate_track` in `sim/track.py` |
| Corner easing | `_ease_tight_corners` in `sim/track.py` |
| Control points, spline, even spacing | `_spline_loop`, `_resample` in `sim/track.py` |
| Heading, radius and validity | `_headings`, `_radius`, `_is_valid` in `sim/track.py` |
| Edges and barriers | `_build` in `sim/track.py` |
| Progress unwrap, grass flag, end rules | `step` in `sim/env.py` |
| Observation scaling | `_obs` in `sim/env.py` |

## Decisions
| Decision | Why |
|---|---|
| Bicycle model, not full dynamics | Forces braking for corners and stays easy to debug and explain |
| Slip added with a separate nose limit | Fast corners should slide, and one parameter controls how much |
| Slip is observed | A real car can estimate it from its motion sensors, and the agent must know when it slides |
| Sensors only, no track map in the observation | The agent must drive from what its lidar sees |
| Barriers one road width past the edges | The sensor still reads something from the grass |
| Rust for the per-step physics, Python for track generation | Step speed, and every seed keeps its track |
| Constants in Rust | One source of truth next to the equations |
| Body-frame update, not a rotation of the velocity | No speed loss in gripping corners, and it matches the textbook equations |

## What is not modelled
Known limits, so nobody is surprised in the defense.
- **No mass or load transfer.** Accelerations are set directly, not computed from forces.
- **No friction circle.** Full throttle or full brake do not reduce sideways grip, which a real tire would do.
- **No tire curve.** Friction is a flat limit, with no peak and fall off.
- **No yaw dynamics.** The yaw rate follows the steering at once. Slip does not turn the car back by itself.
- **Slip acts on the rear axle point.** The front and rear of the car do not slide separately.
- **The car is a point for every check.** The drawn body can overlap a barrier before the check fires. The check also runs after the move, so the car ends past the line by up to one step of travel, about 1.1 m at most. The largest seen in testing is about 0.6 m and the usual value about 0.2 m.
- **Lidar sits on the rear axle**, and it sees the road edge as if it were solid.
- **`info["speed"]` is the forward speed only.** It reads below the true speed during a drift, the on-screen speed uses $\sqrt{v_x^2 + v_y^2}$.
- **First-order integration.** While sliding, the slip update can change the total speed by a factor of at most $\sqrt{1 + \Delta\psi^2}$ per step before friction, which is under 0.4 percent and largest near 8 m/s. The results depend slightly on $\Delta t$, so the time step must not change between training and evaluation.
