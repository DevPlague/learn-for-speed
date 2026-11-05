//! Lidar rays against the road edges and the barriers. See docs/PHYSICS.md.

use std::f64::consts::PI;

pub const N_RAYS: usize = 25;
pub const RANGE: f64 = 30.0; // meters

/// Full circle, denser in front, one ray straight ahead.
pub fn angles() -> [f64; N_RAYS] {
    let half = (N_RAYS / 2) as i32;
    std::array::from_fn(|i| {
        let u = (i as i32 - half) as f64 / (N_RAYS as f64 / 2.0);
        PI * u.signum() * u.abs().powf(1.5)
    })
}

/// Start point, direction vector and length of one edge or barrier segment.
pub struct Segment {
    pub ax: f64,
    pub ay: f64,
    pub ex: f64,
    pub ey: f64,
    pub length: f64,
}

/// Every segment of a closed polyline.
pub fn segments_of(line: &[[f64; 2]], out: &mut Vec<Segment>) {
    for (i, a) in line.iter().enumerate() {
        let b = line[(i + 1) % line.len()];
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        out.push(Segment {
            ax: a[0],
            ay: a[1],
            ex,
            ey,
            length: ex.hypot(ey),
        });
    }
}

/// Distance in meters to the nearest edge or barrier along each ray, capped at RANGE.
pub fn cast(segs: &[Segment], angles: &[f64; N_RAYS], x: f64, y: f64, yaw: f64) -> [f64; N_RAYS] {
    let dirs: [(f64, f64); N_RAYS] = std::array::from_fn(|i| {
        let (s, c) = (yaw + angles[i]).sin_cos();
        (c, s)
    });
    let mut hits = [RANGE; N_RAYS];
    for seg in segs {
        let (aox, aoy) = (seg.ax - x, seg.ay - y);
        let reach = RANGE + seg.length;
        if aox * aox + aoy * aoy >= reach * reach {
            continue; // too far away for any ray to reach it
        }
        let cross_ao_e = aox * seg.ey - aoy * seg.ex;
        for (hit, &(dx, dy)) in hits.iter_mut().zip(&dirs) {
            // solve pos + t*d = a + u*e, parallel rays give NaN or infinity and fail the test
            let denom = dx * seg.ey - dy * seg.ex;
            let t = cross_ao_e / denom;
            let u = (aox * dy - aoy * dx) / denom;
            if t > 0.0 && (0.0..=1.0).contains(&u) && t < *hit {
                *hit = t;
            }
        }
    }
    hits
}
