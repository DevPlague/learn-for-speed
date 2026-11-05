//! Queries along the centerline. See docs/PHYSICS.md.

pub const WINDOW: i64 = 20; // points searched each way around the previous index

pub struct Centerline {
    pub points: Vec<[f64; 2]>,
    pub heading: Vec<f64>,
    pub s: Vec<f64>,
    pub length: f64,
}

impl Centerline {
    /// Index of the nearest centerline point. Searches only near `last` when it is given.
    pub fn locate(&self, x: f64, y: f64, last: Option<i64>, window: i64) -> i64 {
        let n = self.points.len() as i64;
        let dist2 = |i: i64| {
            let p = self.points[i as usize];
            (p[0] - x).powi(2) + (p[1] - y).powi(2)
        };
        let (start, count) = match last {
            Some(last) => (last - window, 2 * window + 1),
            None => (0, n),
        };
        let mut best = start.rem_euclid(n);
        let mut best_d = dist2(best);
        for k in 1..count {
            let i = (start + k).rem_euclid(n);
            let d = dist2(i);
            if d < best_d {
                best = i;
                best_d = d;
            }
        }
        best
    }

    /// Progress in [0, 1) along the lap and signed lateral offset (left is positive).
    pub fn frenet(&self, x: f64, y: f64, idx: i64) -> (f64, f64) {
        let i = idx as usize;
        let (sin_h, cos_h) = self.heading[i].sin_cos();
        let (dx, dy) = (x - self.points[i][0], y - self.points[i][1]);
        let along = dx * cos_h + dy * sin_h;
        let lateral = -dx * sin_h + dy * cos_h;
        (
            (self.s[i] + along).rem_euclid(self.length) / self.length,
            lateral,
        )
    }
}
