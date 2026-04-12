//! Segment pair ensemble clustering for multidomain regions.
//! Simplified port of p7_spensemble.c.

/// A segment pair from a stochastic trace.
#[derive(Debug, Clone)]
pub struct SegmentPair {
    pub i: usize,  // sequence start (1-based)
    pub j: usize,  // sequence end
    pub k: usize,  // HMM start (1-based)
    pub m: usize,  // HMM end
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
}

impl Default for ClusterParams {
    fn default() -> Self {
        ClusterParams {
            min_overlap: 0.8,
            min_posterior: 0.25,
            min_endpointp: 0.02,
            max_diagdiff: 4,
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

    // For each cluster, compute envelope and posterior
    let mut envelopes = Vec::new();
    for (_root, members) in &cluster_map {
        // Count unique traces in this cluster
        let mut trace_set = std::collections::HashSet::new();
        for &idx in members {
            trace_set.insert(segments[idx].trace_idx);
        }
        let posterior = trace_set.len() as f32 / ntraces as f32;

        if posterior < params.min_posterior {
            continue;
        }

        // Compute consensus envelope: widest coordinates
        let mut ienv = usize::MAX;
        let mut jenv = 0;
        let mut kenv = usize::MAX;
        let mut menv = 0;

        for &idx in members {
            ienv = ienv.min(segments[idx].i);
            jenv = jenv.max(segments[idx].j);
            kenv = kenv.min(segments[idx].k);
            menv = menv.max(segments[idx].m);
        }

        envelopes.push(DomainEnvelope {
            ienv,
            jenv,
            kenv,
            menv,
            posterior,
        });
    }

    // Sort by sequence position
    envelopes.sort_by_key(|e| e.ienv);

    // Remove dominated (overlapping) envelopes
    let mut result = Vec::new();
    for env in &envelopes {
        let dominated = result.iter().any(|prev: &DomainEnvelope| {
            let overlap = overlap_fraction(prev.ienv, prev.jenv, env.ienv, env.jenv);
            overlap > params.min_overlap && prev.posterior > env.posterior
        });
        if !dominated {
            result.push(env.clone());
        }
    }

    result
}

/// Check if two segments overlap sufficiently.
fn segments_overlap(a: &SegmentPair, b: &SegmentPair, params: &ClusterParams) -> bool {
    // Sequence overlap
    let seq_ovl = overlap_fraction(a.i, a.j, b.i, b.j);
    if seq_ovl < params.min_overlap {
        return false;
    }

    // HMM overlap
    let hmm_ovl = overlap_fraction(a.k, a.m, b.k, b.m);
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
fn overlap_fraction(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> f32 {
    let overlap_start = a_start.max(b_start);
    let overlap_end = a_end.min(b_end);
    if overlap_start > overlap_end {
        return 0.0;
    }
    let overlap_len = (overlap_end - overlap_start + 1) as f32;
    let a_len = (a_end - a_start + 1) as f32;
    let b_len = (b_end - b_start + 1) as f32;
    overlap_len / a_len.min(b_len)
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
            SegmentPair { i: 10, j: 50, k: 1, m: 40, trace_idx: 0 },
            SegmentPair { i: 11, j: 49, k: 1, m: 40, trace_idx: 1 },
            SegmentPair { i: 10, j: 51, k: 1, m: 40, trace_idx: 2 },
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
            SegmentPair { i: 10, j: 50, k: 1, m: 40, trace_idx: 0 },
            SegmentPair { i: 11, j: 49, k: 1, m: 40, trace_idx: 1 },
            SegmentPair { i: 10, j: 50, k: 1, m: 40, trace_idx: 2 },
            // Domain 2: positions 100-140
            SegmentPair { i: 100, j: 140, k: 1, m: 40, trace_idx: 0 },
            SegmentPair { i: 101, j: 139, k: 1, m: 40, trace_idx: 1 },
            SegmentPair { i: 100, j: 140, k: 1, m: 40, trace_idx: 2 },
        ];
        let params = ClusterParams::default();
        let envs = cluster(&segments, 10, &params);
        assert_eq!(envs.len(), 2);
        assert!(envs[0].ienv <= 11);
        assert!(envs[1].ienv >= 100);
    }
}
