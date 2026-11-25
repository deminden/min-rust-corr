//! First and second nearest neighbours in the transformed plane.
//! Brute force with squared distances, each distance computed once;
//! ties go to the lowest index.

/// Nearest-neighbour results for every point.
#[derive(Debug, Default, Clone)]
pub struct Neighbors {
    /// Index of the nearest neighbour.
    pub nn: Vec<u32>,
    /// Squared distance to the nearest neighbour.
    pub d1: Vec<f64>,
    /// Squared distance to the second nearest neighbour.
    pub d2: Vec<f64>,
}

impl Neighbors {
    /// Compute for points (x_i, y_i); requires at least 3 points.
    pub fn compute(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len();
        self.nn.clear();
        self.nn.resize(n, u32::MAX);
        self.d1.clear();
        self.d1.resize(n, f64::INFINITY);
        self.d2.clear();
        self.d2.resize(n, f64::INFINITY);
        for i in 0..n {
            let (xi, yi) = (x[i], y[i]);
            for j in i + 1..n {
                let (dx, dy) = (x[j] - xi, y[j] - yi);
                let d = dx * dx + dy * dy;
                self.offer(i, j, d);
                self.offer(j, i, d);
            }
        }
    }

    /// Candidates arrive in increasing index order for each point, so a
    /// strict comparison keeps the lowest index on ties.
    #[inline]
    fn offer(&mut self, i: usize, j: usize, d: f64) {
        if d < self.d1[i] {
            self.d2[i] = self.d1[i];
            self.d1[i] = d;
            self.nn[i] = j as u32;
        } else if d < self.d2[i] {
            self.d2[i] = d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let x = [0.0, 1.0, 3.0];
        let y = [0.0; 3];
        let mut nb = Neighbors::default();
        nb.compute(&x, &y);
        assert_eq!(nb.nn, vec![1, 0, 1]);
        assert_eq!(nb.d1, vec![1.0, 1.0, 4.0]);
        assert_eq!(nb.d2, vec![9.0, 4.0, 9.0]);
    }
}
