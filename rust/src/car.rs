//! Bicycle car that grips in slow corners and drifts in fast ones. See docs/PHYSICS.md.
//!
//! State is [x, y, yaw, v, steer, v_lat]. v is the forward speed (v < 0 is reverse) and v_lat
//! the sideways slip speed in the car frame (positive is to the left).

pub const DT: f64 = 0.05;
pub const WHEELBASE: f64 = 2.5;
pub const MAX_STEER: f64 = 0.5; // rad
pub const STEER_RATE: f64 = 2.0; // rad/s
pub const MAX_SPEED: f64 = 20.0;
pub const REVERSE_SPEED: f64 = 5.0;
pub const ACCEL: f64 = 6.0; // m/s^2 at full throttle
pub const BRAKE: f64 = 10.0; // m/s^2 at full brake
pub const DRAG: f64 = 0.05; // 1/s
pub const GRIP: f64 = 8.0; // max sideways acceleration the tires hold, m/s^2
pub const YAW_GRIP: f64 = 14.0; // the nose may turn this hard, past GRIP the car slides
pub const GRASS_DRAG: f64 = 0.8; // extra drag on grass, caps speed near 7 m/s
pub const GRASS_GRIP: f64 = 4.0;

/// Advance one DT. action = [steer, throttle/brake] in [-1, 1].
///
/// Pushing against the motion brakes. From standstill a negative throttle drives backwards.
pub fn step_car(state: &[f64; 6], action: &[f64; 2], on_grass: bool) -> [f64; 6] {
    let [mut x, mut y, mut yaw, v, mut steer, mut v_lat] = *state;
    let steer_cmd = action[0].clamp(-1.0, 1.0) * MAX_STEER;
    steer += (steer_cmd - steer).clamp(-STEER_RATE * DT, STEER_RATE * DT);
    let grip = if on_grass { GRASS_GRIP } else { GRIP };

    let throttle = action[1].clamp(-1.0, 1.0);
    let braking = v * throttle < 0.0;
    let drag = DRAG + if on_grass { GRASS_DRAG } else { 0.0 };
    let force = if braking { BRAKE } else { ACCEL };
    let mut v_new = v + (force * throttle - drag * v) * DT;
    if braking && v_new * v < 0.0 {
        v_new = 0.0; // brakes stop the car, reverse starts on the next step
    }
    v_new = v_new.clamp(-REVERSE_SPEED, MAX_SPEED);

    let mut tan_steer = steer.tan();
    if v_new.abs() > 1e-3 {
        let limit = YAW_GRIP * grip / GRIP * WHEELBASE / (v_new * v_new);
        tan_steer = tan_steer.clamp(-limit, limit);
    }
    let dyaw = v_new * tan_steer / WHEELBASE * DT;

    // body-frame equations of motion, turning moves speed between the two axes of the car
    let v_long = (v_new + dyaw * v_lat).clamp(-REVERSE_SPEED, MAX_SPEED);
    v_lat -= dyaw * v_new;
    // tires pull the slip back to zero, at most at the grip limit
    v_lat -= v_lat.clamp(-grip * DT, grip * DT);

    yaw += dyaw;
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    x += (v_long * cos_yaw - v_lat * sin_yaw) * DT;
    y += (v_long * sin_yaw + v_lat * cos_yaw) * DT;
    [x, y, yaw, v_long, steer, v_lat]
}
