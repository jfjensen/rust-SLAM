// Part 2: the course's log file format (lego_robot.py's LegoLogfile).
//
// One record per line, the first word says what it is:
//
//   M time  lpos ltacho lacc lspeed  rpos rtacho racc rspeed  (3rd motor x4)
//          motor encoders. Absolute ticks: left at word 2, right at word 6.
//   S time count d0 d1 ... d659
//          one LiDAR scan, ranges in mm; 20 or less means "no echo".
//   P time x y
//          reference (true) position of the scanner, integer mm.
//   L C x y diameter
//          a landmark (a cylinder) in the arena.
//   F x y [heading]
//          a filtered pose, written by an algorithm.
//   D C x y x y ...
//          cylinders detected in one scan, in the scanner's frame.
//
// Like LegoLogfile.read, reading several files merges them: each kind of
// record comes from the last file that has any of it. That's how the
// course reads robot4_motors.txt and robot4_scan.txt into one log.

use crate::robot::Pose;
use bevy::math::Vec2;
use std::fmt::{self, Write as _};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotorRecord {
    pub time_ms: u64,
    pub left: i64,
    pub right: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanRecord {
    pub time_ms: u64,
    pub ranges: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceRecord {
    pub time_ms: u64,
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landmark {
    pub position: Vec2,
    pub diameter: f32,
}

/// Everything read from (or to be written to) the log files.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogFile {
    pub motors: Vec<MotorRecord>,
    pub scans: Vec<ScanRecord>,
    pub reference: Vec<ReferenceRecord>,
    pub landmarks: Vec<Landmark>,
    pub filtered: Vec<Pose>,
    pub detected: Vec<Vec<Vec2>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

// --- Part 2a: reading ----------------------------------------------------------

impl LogFile {
    /// The number of time steps: the longest of the per-step lists
    /// (some may be empty), as LegoLogfile.size().
    pub fn len(&self) -> usize {
        [
            self.motors.len(),
            self.scans.len(),
            self.reference.len(),
            self.filtered.len(),
            self.detected.len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Parse one file's text.
    pub fn parse(text: &str) -> Result<LogFile, ParseError> {
        let mut log = LogFile::default();
        for (n, line) in text.lines().enumerate() {
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }
            let err = |message: String| ParseError {
                line: n + 1,
                message,
            };
            let int = |i: usize| -> Result<i64, ParseError> {
                let w = words
                    .get(i)
                    .ok_or_else(|| err(format!("missing word {i}")))?;
                w.parse()
                    .map_err(|_| err(format!("'{w}' is not an integer")))
            };
            let floats = |from: usize| -> Result<Vec<f32>, ParseError> {
                words[from..]
                    .iter()
                    .map(|w| w.parse().map_err(|_| err(format!("'{w}' is not a number"))))
                    .collect()
            };
            match words[0] {
                "M" => log.motors.push(MotorRecord {
                    time_ms: int(1)? as u64,
                    left: int(2)?,
                    right: int(6)?,
                }),
                "S" => {
                    let count = int(2)? as usize;
                    let ranges = (3..3 + count)
                        .map(|i| int(i).map(|v| v.max(0) as u32))
                        .collect::<Result<Vec<u32>, _>>()?;
                    log.scans.push(ScanRecord {
                        time_ms: int(1)? as u64,
                        ranges,
                    });
                }
                "P" => log.reference.push(ReferenceRecord {
                    time_ms: int(1)? as u64,
                    x: int(2)? as i32,
                    y: int(3)? as i32,
                }),
                "L" if words.get(1) == Some(&"C") => {
                    let v = floats(2)?;
                    if v.len() < 3 {
                        return Err(err("L C needs x y diameter".into()));
                    }
                    log.landmarks.push(Landmark {
                        position: Vec2::new(v[0], v[1]),
                        diameter: v[2],
                    });
                }
                "F" => {
                    let v = floats(1)?;
                    if v.len() < 2 {
                        return Err(err("F needs x y [heading]".into()));
                    }
                    log.filtered
                        .push(Pose::new(v[0], v[1], v.get(2).copied().unwrap_or(0.0)));
                }
                "D" if words.get(1) == Some(&"C") => {
                    let v = floats(2)?;
                    log.detected.push(
                        v.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|&[x, y]| Vec2::new(x, y))
                            .collect(),
                    );
                }
                // Records the later units add (E, W, PA, I ...): not yet.
                _ => {}
            }
        }
        Ok(log)
    }

    /// Merge another file into this one: every kind of record the other
    /// file has replaces ours (LegoLogfile.read called twice).
    pub fn merge(&mut self, other: LogFile) {
        fn take<T>(ours: &mut Vec<T>, theirs: Vec<T>) {
            if !theirs.is_empty() {
                *ours = theirs;
            }
        }
        take(&mut self.motors, other.motors);
        take(&mut self.scans, other.scans);
        take(&mut self.reference, other.reference);
        take(&mut self.landmarks, other.landmarks);
        take(&mut self.filtered, other.filtered);
        take(&mut self.detected, other.detected);
    }

    /// Read and merge several files, in order. Missing files are an error.
    pub fn read_files<P: AsRef<Path>>(paths: &[P]) -> Result<LogFile, String> {
        let mut log = LogFile::default();
        for path in paths {
            let path = path.as_ref();
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let one = LogFile::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            log.merge(one);
        }
        Ok(log)
    }

    /// Tick increments per step, as LegoLogfile stores motor_ticks: the
    /// first step's increment is (0, 0).
    pub fn motor_increments(&self) -> Vec<(i64, i64)> {
        (0..self.motors.len())
            .map(|i| self.motor_increment(i))
            .collect()
    }

    /// The tick increment of step i ((0, 0) for the first step, or past
    /// the end). The algorithm plugins use this one step at a time.
    pub fn motor_increment(&self, i: usize) -> (i64, i64) {
        match (
            i.checked_sub(1).and_then(|j| self.motors.get(j)),
            self.motors.get(i),
        ) {
            (Some(a), Some(b)) => (b.left - a.left, b.right - a.right),
            _ => (0, 0),
        }
    }
}

// --- Part 2b: writing ----------------------------------------------------------
//
// One function per file the course uses. The motor record repeats the
// position as the tacho count, and writes the acceleration and speed
// fields the real robot logs (only words 2 and 6 are ever read).

/// Fixed motor fields, as in the course's robot4_motors.txt.
const MOTOR_ACCELERATION: i64 = 3000;
const THIRD_MOTOR: &str = "0 0 6000 0";

pub fn motors_text(motors: &[MotorRecord]) -> String {
    let mut s = String::new();
    let mut last: Option<&MotorRecord> = None;
    for m in motors {
        // Wheel speeds in degrees per second (one tick is one degree).
        let (ls, rs) = match last {
            Some(p) if m.time_ms > p.time_ms => {
                let dt = (m.time_ms - p.time_ms) as f64 / 1000.0;
                (
                    ((m.left - p.left) as f64 / dt).round() as i64,
                    ((m.right - p.right) as f64 / dt).round() as i64,
                )
            }
            _ => (0, 0),
        };
        let _ = writeln!(
            s,
            "M {} {} {} {MOTOR_ACCELERATION} {ls} {} {} {MOTOR_ACCELERATION} {rs} {THIRD_MOTOR}",
            m.time_ms, m.left, m.left, m.right, m.right
        );
        last = Some(m);
    }
    s
}

pub fn scans_text(scans: &[ScanRecord]) -> String {
    let mut s = String::new();
    for scan in scans {
        let _ = write!(s, "S {} {}", scan.time_ms, scan.ranges.len());
        for r in &scan.ranges {
            let _ = write!(s, " {r}");
        }
        s.push('\n');
    }
    s
}

pub fn reference_text(reference: &[ReferenceRecord]) -> String {
    let mut s = String::new();
    for p in reference {
        let _ = writeln!(s, "P {} {} {}", p.time_ms, p.x, p.y);
    }
    s
}

pub fn landmarks_text(landmarks: &[Landmark]) -> String {
    let mut s = String::new();
    for l in landmarks {
        let _ = writeln!(
            s,
            "L C {:.1}\t{:.1}\t{:.1}",
            l.position.x, l.position.y, l.diameter
        );
    }
    s
}

/// poses_from_ticks.txt: "F x y heading".
pub fn filtered_text(poses: &[Pose]) -> String {
    poses.iter().map(pose_line).collect()
}

/// cylinders.txt: "D C x y x y ...", one line per scan, also when empty.
pub fn detected_text(detected: &[Vec<Vec2>]) -> String {
    detected
        .iter()
        .map(|scan| points_line("D C", scan))
        .collect()
}

/// One F record, with its newline.
pub fn pose_line(p: &Pose) -> String {
    format!("F {:.6} {:.6} {:.6}\n", p.x, p.y, p.heading)
}

/// One point-list record, "D C x y x y ..." or "W C ..." (the course's
/// write_cylinders), with its newline. Also when there are no points.
pub fn points_line(tag: &str, points: &[Vec2]) -> String {
    let mut s = String::from(tag);
    for c in points {
        let _ = write!(s, " {:.1} {:.1}", c.x, c.y);
    }
    s.push('\n');
    s
}

/// The log file names, after the course's, with a prefix ("data/generated" gives
/// data/generated_motors.txt and so on).
pub struct LogPaths {
    pub motors: String,
    pub scan: String,
    pub reference: String,
    pub landmarks: String,
}

impl LogPaths {
    pub fn with_prefix(prefix: &str) -> LogPaths {
        // The landmarks file has no prefix: it describes the arena.
        let dir = Path::new(prefix)
            .parent()
            .map(|d| d.to_string_lossy().into_owned())
            .unwrap_or_default();
        let landmarks = if dir.is_empty() {
            "robot_arena_landmarks.txt".to_string()
        } else {
            format!("{dir}/robot_arena_landmarks.txt")
        };
        LogPaths {
            motors: format!("{prefix}_motors.txt"),
            scan: format!("{prefix}_scan.txt"),
            reference: format!("{prefix}_reference.txt"),
            landmarks,
        }
    }

    pub fn all(&self) -> [&str; 4] {
        [&self.motors, &self.scan, &self.reference, &self.landmarks]
    }

    pub fn write(&self, log: &LogFile) -> std::io::Result<()> {
        if let Some(dir) = Path::new(&self.motors).parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&self.motors, motors_text(&log.motors))?;
        std::fs::write(&self.scan, scans_text(&log.scans))?;
        std::fs::write(&self.reference, reference_text(&log.reference))?;
        std::fs::write(&self.landmarks, landmarks_text(&log.landmarks))
    }

    pub fn read(&self) -> Result<LogFile, String> {
        LogFile::read_files(&self.all())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first lines of the course's robot4_motors.txt, word for word.
    const COURSE_MOTORS: &str = "\
M 204 20795 20795 3000 0 16067 16066 3000 0 0 0 6000 0
M 524 20795 20794 3000 0 16067 16066 3000 0 0 0 6000 0
M 735 20800 20795 3000 0 16075 16066 3000 0 0 0 6000 0
";

    #[test]
    fn reads_the_course_motor_format() {
        let log = LogFile::parse(COURSE_MOTORS).unwrap();
        assert_eq!(log.motors.len(), 3);
        // Word 2 is the left position, word 6 the right one (not word 3,
        // the tacho count, which differs by one in the second line).
        assert_eq!(log.motors[1].left, 20795);
        assert_eq!(log.motors[0].right, 16067);
        assert_eq!(log.motor_increments(), vec![(0, 0), (0, 0), (5, 8)]);
    }

    #[test]
    fn scan_record_has_a_count() {
        let log = LogFile::parse("S 315 4 189 186 0 2000\n").unwrap();
        assert_eq!(log.scans[0].ranges, vec![189, 186, 0, 2000]);
        assert_eq!(log.scans[0].time_ms, 315);
    }

    #[test]
    fn what_we_write_we_read_back() {
        let log = LogFile {
            motors: vec![
                MotorRecord {
                    time_ms: 0,
                    left: 10,
                    right: 20,
                },
                MotorRecord {
                    time_ms: 200,
                    left: 110,
                    right: 90,
                },
            ],
            scans: vec![ScanRecord {
                time_ms: 0,
                ranges: vec![1, 2, 3],
            }],
            reference: vec![ReferenceRecord {
                time_ms: 0,
                x: 5,
                y: -6,
            }],
            landmarks: vec![Landmark {
                position: Vec2::new(1.5, 2.0),
                diameter: 90.0,
            }],
            filtered: vec![Pose::new(1.0, 2.0, 0.5)],
            detected: vec![vec![], vec![Vec2::new(100.0, -20.5)]],
        };
        let text = [
            motors_text(&log.motors),
            scans_text(&log.scans),
            reference_text(&log.reference),
            landmarks_text(&log.landmarks),
            filtered_text(&log.filtered),
            detected_text(&log.detected),
        ]
        .concat();
        assert_eq!(LogFile::parse(&text).unwrap(), log);
        // Speeds in the motor record: 100 ticks in 0.2 s is 500 per second.
        assert!(text.contains("M 200 110 110 3000 500 90 90 3000 350"));
    }

    #[test]
    fn merging_replaces_only_what_the_new_file_has() {
        let mut log = LogFile::parse(COURSE_MOTORS).unwrap();
        log.merge(LogFile::parse("P 1 2 3\n").unwrap());
        assert_eq!(log.motors.len(), 3);
        assert_eq!(log.reference.len(), 1);
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn bad_numbers_say_where() {
        let e = LogFile::parse("P 1 2 3\nP 1 x 3\n").unwrap_err();
        assert_eq!(e.line, 2);
    }

    #[test]
    fn landmarks_file_sits_next_to_the_logs() {
        let p = LogPaths::with_prefix("data/generated");
        assert_eq!(p.motors, "data/generated_motors.txt");
        assert_eq!(p.landmarks, "data/robot_arena_landmarks.txt");
        assert_eq!(
            LogPaths::with_prefix("live").landmarks,
            "robot_arena_landmarks.txt"
        );
    }
}
