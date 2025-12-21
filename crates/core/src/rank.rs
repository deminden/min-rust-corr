use ndarray::Array1;

pub fn rank_data(data: &Array1<f64>) -> Array1<f64> {
    let mut indexed_data: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; data.len()];
    let mut i = 0;
    while i < data.len() {
        let mut j = i;
        while j < data.len() - 1 && indexed_data[j].1 == indexed_data[j + 1].1 {
            j += 1;
        }
        let rank = (i + 1..=j + 1).sum::<usize>() as f64 / (j - i + 1) as f64;
        for k in i..=j {
            ranks[indexed_data[k].0] = rank;
        }
        i = j + 1;
    }
    Array1::from(ranks)
}
