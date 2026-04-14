//! Segment pair ensemble clustering for multidomain regions.
//! Simplified port of p7_spensemble.c.

/// A segment pair from a stochastic trace.
#[derive(Debug, Clone)]
pub struct SegmentPair {
    pub i: usize,         // sequence start (1-based)
    pub j: usize,         // sequence end
    pub k: usize,         // HMM start (1-based)
    pub m: usize,         // HMM end
    pub trace_idx: usize, // which trace this came from
}

/// Result of clustering: a significant domain envelope.
#[derive(Debug, Clone)]
pub struct DomainEnvelope {
    pub ienv: usize,
    pub jenv: usize,
    pub kenv: usize,
    pub menv: usize,
    pub posterior: f32, // fraction of traces that contain this domain
}

/// Clustering parameters.
pub struct ClusterParams {
    pub min_overlap: f32,    // minimum overlap fraction (default 0.8)
    pub min_posterior: f32,  // minimum posterior for significance (default 0.25)
    pub min_endpointp: f32,  // minimum endpoint probability (default 0.02)
    pub max_diagdiff: usize, // maximum diagonal difference (default 4)
    pub of_smaller: bool,    // use smaller segment as overlap denominator
}

impl Default for ClusterParams {
    fn default() -> Self {
        ClusterParams {
            min_overlap: 0.8,
            min_posterior: 0.25,
            min_endpointp: 0.02,
            max_diagdiff: 4,
            of_smaller: true,
        }
    }
}

/// Cluster segment pairs into domain envelopes.
/// `segments` contains all segment pairs from N stochastic traces.
/// `ntraces` is the total number of traces sampled.
/// Returns significant domain envelopes.
pub fn cluster(
    segments: &[SegmentPair],
    ntraces: usize,
    params: &ClusterParams,
) -> Vec<DomainEnvelope> {
    if segments.is_empty() || ntraces == 0 {
        return Vec::new();
    }

    let n = segments.len();

    // Single-linkage clustering via union-find
    let mut parent: Vec<usize> = (0..n).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if segments_overlap(&segments[i], &segments[j], params) {
                union(&mut parent, i, j);
            }
        }
    }

    // Collect clusters
    let mut cluster_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&parent, i);
        cluster_map.entry(root).or_default().push(i);
    }

    // For each cluster, compute posterior and C-style consensus endpoints.
    let mut envelopes = Vec::new();
    for (_root, members) in &cluster_map {
        // Count unique traces in this cluster.
        let mut trace_set = std::collections::HashSet::new();
        for &idx in members {
            trace_set.insert(segments[idx].trace_idx);
        }
        let posterior = trace_set.len() as f32 / ntraces as f32;

        if posterior < params.min_posterior {
            continue;
        }

        let mut imin = usize::MAX;
        let mut imax = 0usize;
        let mut jmin = usize::MAX;
        let mut jmax = 0usize;
        let mut kmin = usize::MAX;
        let mut kmax = 0usize;
        let mut mmin = usize::MAX;
        let mut mmax = 0usize;

        for &idx in members {
            imin = imin.min(segments[idx].i);
            imax = imax.max(segments[idx].i);
            jmin = jmin.min(segments[idx].j);
            jmax = jmax.max(segments[idx].j);
            kmin = kmin.min(segments[idx].k);
            kmax = kmax.max(segments[idx].k);
            mmin = mmin.min(segments[idx].m);
            mmax = mmax.max(segments[idx].m);
        }

        let epc_threshold = ((trace_set.len() as f32) * params.min_endpointp).ceil() as usize;
        let epc_threshold = epc_threshold.max(1);

        let best_i = left_endpoint(members, segments, |s| s.i, imin, imax, epc_threshold);
        let best_k = left_endpoint(members, segments, |s| s.k, kmin, kmax, epc_threshold);
        let best_j = right_endpoint(members, segments, |s| s.j, jmin, jmax, epc_threshold);
        let best_m = right_endpoint(members, segments, |s| s.m, mmin, mmax, epc_threshold);

        if best_i > best_j || best_k > best_m {
            continue;
        }

        envelopes.push(DomainEnvelope {
            ienv: best_i,
            jenv: best_j,
            kenv: best_k,
            menv: best_m,
            posterior,
        });
    }

    // Sort by sequence position
    envelopes.sort_by_key(|e| e.ienv);

    // Remove dominated domains relative to sequence coords, matching
    // region_trace_ensemble() after p7_spensemble_Cluster().
    let mut dominated = vec![false; envelopes.len()];
    for d in 0..envelopes.len() {
        for d2 in (d + 1)..envelopes.len() {
            let nov = envelopes[d].jenv.min(envelopes[d2].jenv) as isize
                - envelopes[d].ienv.max(envelopes[d2].ienv) as isize
                + 1;
            if nov == 0 {
                break;
            }
            let n = (envelopes[d].jenv - envelopes[d].ienv + 1)
                .min(envelopes[d2].jenv - envelopes[d2].ienv + 1);
            if (nov as f32) / (n as f32) >= 0.8 {
                if envelopes[d].posterior > envelopes[d2].posterior {
                    dominated[d2] = true;
                } else {
                    dominated[d] = true;
                }
            }
        }
    }

    envelopes
        .into_iter()
        .enumerate()
        .filter_map(|(idx, env)| if dominated[idx] { None } else { Some(env) })
        .collect()
}

/// Check if two segments overlap sufficiently.
fn segments_overlap(a: &SegmentPair, b: &SegmentPair, params: &ClusterParams) -> bool {
    // Sequence overlap
    let seq_ovl = overlap_fraction(a.i, a.j, b.i, b.j, true, params.of_smaller);
    if seq_ovl < params.min_overlap {
        return false;
    }

    // HMM overlap
    let hmm_ovl = overlap_fraction(a.k, a.m, b.k, b.m, false, params.of_smaller);
    if hmm_ovl < params.min_overlap {
        return false;
    }

    // Diagonal proximity
    let diag_a_start = (a.i as i64) - (a.k as i64);
    let diag_b_start = (b.i as i64) - (b.k as i64);
    let diag_a_end = (a.j as i64) - (a.m as i64);
    let diag_b_end = (b.j as i64) - (b.m as i64);

    let start_diff = (diag_a_start - diag_b_start).unsigned_abs() as usize;
    let end_diff = (diag_a_end - diag_b_end).unsigned_abs() as usize;

    start_diff <= params.max_diagdiff || end_diff <= params.max_diagdiff
}

/// Compute overlap fraction between two intervals.
fn overlap_fraction(
    a_start: usize,
    a_end: usize,
    b_start: usize,
    b_end: usize,
    include_endpoint: bool,
    of_smaller: bool,
) -> f32 {
    let endpoint = if include_endpoint { 1isize } else { 0isize };
    let overlap_len = a_end.min(b_end) as isize - a_start.max(b_start) as isize + endpoint;
    if overlap_len <= 0 {
        return 0.0;
    }
    let a_len = a_end - a_start + 1;
    let b_len = b_end - b_start + 1;
    let denom = if of_smaller {
        a_len.min(b_len)
    } else {
        a_len.max(b_len)
    };
    overlap_len as f32 / denom as f32
}

fn left_endpoint<F>(
    members: &[usize],
    segments: &[SegmentPair],
    coord: F,
    min_coord: usize,
    max_coord: usize,
    threshold: usize,
) -> usize
where
    F: Fn(&SegmentPair) -> usize,
{
    let mut counts = vec![0usize; max_coord - min_coord + 1];
    for &idx in members {
        counts[coord(&segments[idx]) - min_coord] += 1;
    }
    for (offset, &count) in counts.iter().enumerate() {
        if count >= threshold {
            return min_coord + offset;
        }
    }
    min_coord + argmax_first(&counts)
}

fn right_endpoint<F>(
    members: &[usize],
    segments: &[SegmentPair],
    coord: F,
    min_coord: usize,
    max_coord: usize,
    threshold: usize,
) -> usize
where
    F: Fn(&SegmentPair) -> usize,
{
    let mut counts = vec![0usize; max_coord - min_coord + 1];
    for &idx in members {
        counts[coord(&segments[idx]) - min_coord] += 1;
    }
    for offset in (0..counts.len()).rev() {
        if counts[offset] >= threshold {
            return min_coord + offset;
        }
    }
    min_coord + argmax_first(&counts)
}

fn argmax_first(values: &[usize]) -> usize {
    let mut best = 0usize;
    let mut best_val = values[0];
    for (idx, &val) in values.iter().enumerate().skip(1) {
        if val > best_val {
            best = idx;
            best_val = val;
        }
    }
    best
}

/// Union-find: find root.
fn find(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

/// Union-find: merge sets.
fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_domain() {
        let segments = vec![
            SegmentPair {
                i: 10,
                j: 50,
                k: 1,
                m: 40,
                trace_idx: 0,
            },
            SegmentPair {
                i: 11,
                j: 49,
                k: 1,
                m: 40,
                trace_idx: 1,
            },
            SegmentPair {
                i: 10,
                j: 51,
                k: 1,
                m: 40,
                trace_idx: 2,
            },
        ];
        let params = ClusterParams::default();
        let envs = cluster(&segments, 10, &params);
        assert_eq!(envs.len(), 1);
        assert_eq!(envs[0].ienv, 10);
        assert!(envs[0].posterior >= 0.25);
    }

    #[test]
    fn test_two_domains() {
        let segments = vec![
            // Domain 1: positions 10-50
            SegmentPair {
                i: 10,
                j: 50,
                k: 1,
                m: 40,
                trace_idx: 0,
            },
            SegmentPair {
                i: 11,
                j: 49,
                k: 1,
                m: 40,
                trace_idx: 1,
            },
            SegmentPair {
                i: 10,
                j: 50,
                k: 1,
                m: 40,
                trace_idx: 2,
            },
            // Domain 2: positions 100-140
            SegmentPair {
                i: 100,
                j: 140,
                k: 1,
                m: 40,
                trace_idx: 0,
            },
            SegmentPair {
                i: 101,
                j: 139,
                k: 1,
                m: 40,
                trace_idx: 1,
            },
            SegmentPair {
                i: 100,
                j: 140,
                k: 1,
                m: 40,
                trace_idx: 2,
            },
        ];
        let params = ClusterParams::default();
        let envs = cluster(&segments, 10, &params);
        assert_eq!(envs.len(), 2);
        assert!(envs[0].ienv <= 11);
        assert!(envs[1].ienv >= 100);
    }
}
