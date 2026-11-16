//! Bicycle car that grips in slow corners and drifts in fast ones. State is [x, y, yaw, v, steer, v_lat].

//Time
pub const DT: f64 = 0.05; // steps per second

//Geometry
pub const WHEELBASE: f64 = 2.5; //distance betweem car front axis and back axis.

//Direction
pub const MAX_STEER: f64 = 0.5; // rad. Max angle of the wheels
pub const STEER_RATE: f64 = 2.0; // rad/s. Max angular speed.


// Spped
pub const MAX_SPEED: f64 = 20.0;
pub const REVERSE_SPEED: f64 = 5.0;
pub const ACCEL: f64 = 6.0; // m/s^2 at full throttle
pub const BRAKE: f64 = 10.0; // m/s^2 at full brake
pub const DRAG: f64 = 0.05; // 1/s. Ground friction

// Grip and drift
pub const GRIP: f64 = 8.0; // max sideways acceleration the wheels resists, m/s^2
pub const YAW_GRIP: f64 = 14.0; // the limit to how fast the nose of the car can turn.

// Grass
pub const GRASS_DRAG: f64 = 0.8; // grass friction
pub const GRASS_GRIP: f64 = 4.0; // less grip on grass.


pub fn step_car(state: &[f64; 6], action: &[f64; 2], on_grass: bool) -> [f64; 6] {
    let [mut x, mut y, mut yaw, v, mut steer, mut v_lat] = *state;

    // next change of direction (limited by the time of the step and steer rate)
    let steer_cmd = action[0].clamp(-1.0, 1.0) * MAX_STEER;
    steer += (steer_cmd - steer).clamp(-STEER_RATE * DT, STEER_RATE * DT);

    // friction and grip depending of the area
    let grip = if on_grass { GRASS_GRIP } else { GRIP };
    let drag = DRAG + if on_grass { GRASS_DRAG } else { 0.0 };


    // acceleration
    let throttle = action[1].clamp(-1.0, 1.0);
    let braking = v * throttle < 0.0;
    let force = if braking { BRAKE } else { ACCEL };
    let mut v_new = v + (force * throttle - drag * v) * DT;
    if braking && v_new * v < 0.0 {
        v_new = 0.0; // brakes stop the car, reverse starts on the next step
    }
    v_new = v_new.clamp(-REVERSE_SPEED, MAX_SPEED);


    // Rotation depending of the grip, wheelbase, yaw grip and speed
    let mut tan_steer = steer.tan();
    if v_new.abs() > 1e-3 {
        let limit = YAW_GRIP * grip / GRIP * WHEELBASE / (v_new * v_new);
        tan_steer = tan_steer.clamp(-limit, limit);
    }
    let dyaw = v_new * tan_steer / WHEELBASE * DT;


    // longitudinal speed and drift (latitude speed). Speed will vary in each axis (vlat losts speed with more speed)
    let v_long = (v_new + dyaw * v_lat).clamp(-REVERSE_SPEED, MAX_SPEED);
    v_lat -= dyaw * v_new;
    v_lat -= v_lat.clamp(-grip * DT, grip * DT);

    yaw += dyaw;
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    x += (v_long * cos_yaw - v_lat * sin_yaw) * DT;
    y += (v_long * sin_yaw + v_lat * cos_yaw) * DT;
    [x, y, yaw, v_long, steer, v_lat] // new car state
}
