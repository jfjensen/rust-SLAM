// Unit A, part 2: finding the cylinders in a scan (the course's slam_03).
//
//   1. The derivative of the ranges: (scan[i+1] - scan[i-1]) / 2, and 0
//      where a neighbour has no echo.
//   2. A cylinder stands in front of the wall: a big negative jump (left
//      edge, coming closer) then a big positive one (right edge). Average
//      the beam indices and the ranges in between.
//   3. To the scanner's frame: angle from the average beam, distance =
//      average range + an offset (the ranges hit the FRONT of the
//      cylinder, not its centre).

use crate::core::{AvSet, RobotLog};
use crate::robot::beam_index_to_angle;
use bevy::math::Vec2;
use bevy::prelude::*;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct DetectorParams {
    /// Ranges at or below this are "no echo".
    pub min_valid_distance: f32,
    /// A derivative bigger than this is an edge.
    pub depth_jump: f32,
    /// Front surface to centre. The mean range over a cylinder of radius
    /// r is about r * pi/4 short of its centre: 35 mm for our 90 mm ones.
    /// (The course's 90 was found by experiment for its real cylinders.)
    pub cylinder_offset: f32,
}

impl Default for DetectorParams {
    fn default() -> DetectorParams {
        DetectorParams {
            min_valid_distance: 20.0,
            depth_jump: 100.0,
            cylinder_offset: 35.0,
        }
    }
}

/// One cylinder found in a scan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
    /// Average beam index (fractional).
    pub ray: f32,
    /// Average range of its beams, mm.
    pub depth: f32,
    /// Its centre in the scanner's frame (x ahead, y left).
    pub local: Vec2,
}

pub fn compute_derivative(scan: &[u32], min_dist: f32) -> Vec<f32> {
    let n = scan.len();
    let mut jumps = vec![0.0; n];
    for i in 1..n.saturating_sub(1) {
        let (l, r) = (scan[i - 1] as f32, scan[i + 1] as f32);
        if l > min_dist && r > min_dist {
            jumps[i] = (r - l) / 2.0;
        }
    }
    jumps
}

/// (average ray, average depth) for each cylinder.
pub fn find_cylinders(scan: &[u32], derivative: &[f32], p: &DetectorParams) -> Vec<(f32, f32)> {
    let mut found = Vec::new();
    // Some((sum of rays, sum of depths, count)) while between two edges.
    let mut current: Option<(f32, f32, u32)> = None;
    // A step in the ranges between beams k and k+1 shows in the derivative
    // at BOTH k and k+1. Left edge: k is still the wall, so each new left
    // edge starts again (and if we were already on a cylinder, something
    // even closer stands in front of it). Right edge: k is the cylinder's
    // last beam, so count it before closing. (The course closes first,
    // which drops the last beam and pulls the average ray to the left.)
    for (i, (&range, &d)) in scan.iter().zip(derivative).enumerate() {
        if d < -p.depth_jump {
            current = Some((0.0, 0.0, 0));
        }
        if range as f32 <= p.min_valid_distance {
            // No echo inside a cylinder: we can't trust it, drop it.
            current = None;
        } else if let Some((rays, depths, n)) = &mut current {
            *rays += i as f32;
            *depths += range as f32;
            *n += 1;
        }
        if d > p.depth_jump
            && let Some((rays, depths, n)) = current.take()
        {
            found.push((rays / n as f32, depths / n as f32));
        }
    }
    found
}

/// Polar (beam, depth) to the scanner's Cartesian frame.
pub fn to_cartesian(ray: f32, depth: f32, p: &DetectorParams) -> Vec2 {
    Vec2::from_angle(beam_index_to_angle(ray)) * (depth + p.cylinder_offset)
}

/// All three steps for one scan.
pub fn detect(scan: &[u32], p: &DetectorParams) -> Vec<Cylinder> {
    let derivative = compute_derivative(scan, p.min_valid_distance);
    find_cylinders(scan, &derivative, p)
        .into_iter()
        .map(|(ray, depth)| Cylinder {
            ray,
            depth,
            local: to_cartesian(ray, depth, p),
        })
        .collect()
}

// --- The plugin ----------------------------------------------------------------
//
// Same pattern as MotionModelPlugin: one entry per scan, only new scans
// each frame, start over for a new log.

#[derive(Resource, Default, Debug)]
pub struct DetectedCylinders {
    pub per_scan: Vec<Vec<Cylinder>>,
    generation: u64,
}

#[derive(Default)]
pub struct CylinderDetectionPlugin;

impl Plugin for CylinderDetectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DetectorParams>()
            .init_resource::<DetectedCylinders>()
            .add_systems(Update, update_detections.in_set(AvSet::Algorithms));
    }
}

fn update_detections(
    log: Res<RobotLog>,
    params: Res<DetectorParams>,
    mut found: ResMut<DetectedCylinders>,
) {
    if found.generation != log.generation || params.is_changed() {
        found.per_scan.clear();
        found.generation = log.generation;
    }
    let scans = &log.log.scans;
    if found.per_scan.len() >= scans.len() {
        return;
    }
    let from = found.per_scan.len();
    for scan in &scans[from..] {
        found.per_scan.push(detect(&scan.ranges, &params));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{Simulator, SynthConfig};
    use crate::world::Arena;

    #[test]
    fn a_dip_in_the_wall_is_one_cylinder() {
        // Wall at 1000, a cylinder at 500 on beams 4 to 6, a dropout at 9.
        let scan = [
            1000, 1000, 1000, 1000, 500, 510, 500, 1000, 1000, 0, 1000, 1000,
        ];
        let p = DetectorParams::default();
        let d = compute_derivative(&scan, p.min_valid_distance);
        assert_eq!(d[3], -250.0);
        // Next to the dropout there's no derivative, so no false edge.
        assert_eq!(d[8], 0.0);
        assert_eq!(d[10], 0.0);
        let found = find_cylinders(&scan, &d, &p);
        assert_eq!(found.len(), 1);
        let (ray, depth) = found[0];
        assert!((ray - 5.0).abs() < 0.01, "{ray}");
        assert!((depth - 503.3).abs() < 0.1, "{depth}");
    }

    #[test]
    fn dropout_inside_a_cylinder_drops_it() {
        let scan = [1000, 1000, 1000, 500, 0, 500, 1000, 1000];
        let p = DetectorParams::default();
        let d = compute_derivative(&scan, p.min_valid_distance);
        assert!(find_cylinders(&scan, &d, &p).is_empty());
    }

    #[test]
    fn beam_330_is_straight_ahead() {
        let p = DetectorParams::default();
        let c = to_cartesian(330.0, 1000.0 - p.cylinder_offset, &p);
        assert!((c.x - 1000.0).abs() < 1e-3);
        assert!(c.y.abs() < 1e-3);
    }

    /// In the synthetic arena, every detection is close to a real
    /// landmark, and most visible landmarks are found.
    #[test]
    fn detections_land_on_the_landmarks() {
        let p = DetectorParams::default();
        let arena = Arena::course();
        let mut sim = Simulator::new(SynthConfig::REALISTIC, arena.clone(), 5);
        let mut pilot = crate::synth::Autopilot::default();
        let mut errors = Vec::new();
        for _ in 0..150 {
            let (l, r) = pilot.command(sim.centre());
            sim.advance(l, r);
            let pose = sim.scanner_pose();
            for c in detect(&sim.records().scan.ranges, &p) {
                let world = pose.to_world(c.local);
                let nearest = arena
                    .landmarks
                    .iter()
                    .map(|l| l.position.distance(world))
                    .fold(f32::INFINITY, f32::min);
                errors.push(nearest);
            }
        }
        errors.sort_by(f32::total_cmp);
        let median = errors[errors.len() / 2];
        assert!(errors.len() > 300, "only {} detections", errors.len());
        assert!(median < 15.0, "median error {median}");
        // A few false ones are fine (as in the course), most are good.
        let good = errors.iter().filter(|&&e| e < 50.0).count();
        assert!(good * 10 >= errors.len() * 9, "{good} of {}", errors.len());
    }
}
