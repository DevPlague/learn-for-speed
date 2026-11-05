//! Python bindings of the per-step physics. One `TrackCore.step` call per env step.

mod car;
mod lidar;
mod track;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn fixed<const N: usize>(array: &PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<[f64; N]> {
    array
        .as_slice()?
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must have {N} values")))
}

fn points(array: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<[f64; 2]>> {
    let view = array.as_array();
    if view.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, 2)"
        )));
    }
    Ok(view.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

/// Advance the car one step without any track.
#[pyfunction]
#[pyo3(signature = (state, action, on_grass=false))]
fn step_car<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    action: PyReadonlyArray1<'py, f64>,
    on_grass: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let next = car::step_car(
        &fixed(&state, "state")?,
        &fixed(&action, "action")?,
        on_grass,
    );
    Ok(PyArray1::from_slice(py, &next))
}

/// One track held by Rust, built once at reset. Holds no car state.
#[pyclass(frozen)]
struct TrackCore {
    center: track::Centerline,
    segs: Vec<lidar::Segment>,
    angles: [f64; lidar::N_RAYS],
}

#[pymethods]
impl TrackCore {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        centerline: PyReadonlyArray2<'_, f64>,
        heading: PyReadonlyArray1<'_, f64>,
        s: PyReadonlyArray1<'_, f64>,
        length: f64,
        left: PyReadonlyArray2<'_, f64>,
        right: PyReadonlyArray2<'_, f64>,
        left_wall: PyReadonlyArray2<'_, f64>,
        right_wall: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let mut segs = Vec::new();
        for (line, name) in [
            (&left, "left"),
            (&right, "right"),
            (&left_wall, "left_wall"),
            (&right_wall, "right_wall"),
        ] {
            lidar::segments_of(&points(line, name)?, &mut segs);
        }
        Ok(Self {
            center: track::Centerline {
                points: points(&centerline, "centerline")?,
                heading: heading.as_slice()?.to_vec(),
                s: s.as_slice()?.to_vec(),
                length,
            },
            segs,
            angles: lidar::angles(),
        })
    }

    /// One env step, returns (state, hits, idx, progress fraction, lateral offset).
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, f64>,
        action: PyReadonlyArray1<'py, f64>,
        idx: i64,
        on_grass: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        i64,
        f64,
        f64,
    )> {
        let next = car::step_car(
            &fixed(&state, "state")?,
            &fixed(&action, "action")?,
            on_grass,
        );
        let idx = self
            .center
            .locate(next[0], next[1], Some(idx), track::WINDOW);
        let (frac, lateral) = self.center.frenet(next[0], next[1], idx);
        let hits = lidar::cast(&self.segs, &self.angles, next[0], next[1], next[2]);
        Ok((
            PyArray1::from_slice(py, &next),
            PyArray1::from_slice(py, &hits),
            idx,
            frac,
            lateral,
        ))
    }

    fn cast<'py>(&self, py: Python<'py>, x: f64, y: f64, yaw: f64) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &lidar::cast(&self.segs, &self.angles, x, y, yaw))
    }

    #[pyo3(signature = (x, y, last=None, window=track::WINDOW))]
    fn locate(&self, x: f64, y: f64, last: Option<i64>, window: i64) -> i64 {
        self.center.locate(x, y, last, window)
    }

    fn frenet(&self, x: f64, y: f64, idx: i64) -> (f64, f64) {
        self.center.frenet(x, y, idx)
    }
}

#[pymodule]
fn racing_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DT", car::DT)?;
    m.add("WHEELBASE", car::WHEELBASE)?;
    m.add("MAX_STEER", car::MAX_STEER)?;
    m.add("STEER_RATE", car::STEER_RATE)?;
    m.add("MAX_SPEED", car::MAX_SPEED)?;
    m.add("REVERSE_SPEED", car::REVERSE_SPEED)?;
    m.add("ACCEL", car::ACCEL)?;
    m.add("BRAKE", car::BRAKE)?;
    m.add("DRAG", car::DRAG)?;
    m.add("GRIP", car::GRIP)?;
    m.add("YAW_GRIP", car::YAW_GRIP)?;
    m.add("GRASS_DRAG", car::GRASS_DRAG)?;
    m.add("GRASS_GRIP", car::GRASS_GRIP)?;
    m.add("N_RAYS", lidar::N_RAYS)?;
    m.add("RANGE", lidar::RANGE)?;
    let angles = PyArray1::from_slice(m.py(), &lidar::angles());
    angles.readwrite().make_nonwriteable();
    m.add("ANGLES", angles)?;
    m.add_function(wrap_pyfunction!(step_car, m)?)?;
    m.add_class::<TrackCore>()?;
    Ok(())
}
