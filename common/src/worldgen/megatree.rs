use crate::{
    dodeca::{Side, SideSet, Vertex},
    graph::{Graph, NodeId},
    math::{MPoint, MVector},
    node::VoxelData,
    world::Material,
};
use rand::{RngExt, SeedableRng};
use rand_distr::Normal;
use rand_pcg::Pcg64Mcg;
use std::sync::LazyLock;

use super::{EnviroFactors, NodeStateKind};
use super::{VoxelCoords, index, line::LineSegment, voxel_center};

/// Branch radius, in absolute hyperbolic distance units.
const BRANCH_RADIUS: f32 = 0.2;
/// Radius of the leaves ball at a terminal node.
const LEAVES_RADIUS: f32 = 0.5;
/// Initial growth budget for a newly seeded tree, in Sky propagation steps.
const INITIAL_REMAINING_GROWTH: f32 = 6.5;
/// Random variation in the growth budget consumed by each Sky node.
const GROWTH_COST_STANDARD_DEVIATION: f32 = 0.3;
/// Value mixed into each node's hash to seed generation. Chosen randomly.
const SEED: u64 = 13334231312061724180;

/// Megatree state for one graph node.
pub struct MegatreeState {
    tree: Option<MegatreeNode>,
    diagonal: Option<Box<CarrierState>>,
}

/// Per-node state aggregating diagonal carriers, separate from any tree node they may create.
///
/// `outgoing` relays signals onward, `segments` renders the line through this node, and
/// `has_children` marks branches originated here rather than signals merely passing through.
struct CarrierState {
    outgoing: Vec<(Side, CarrierSignal)>,
    segments: Vec<LineSegment>,
    has_children: bool,
}

type SignalKey = (NodeId, u16, u16);

impl MegatreeState {
    pub fn new(graph: &Graph, node: NodeId, kind: NodeStateKind, enviro: EnviroFactors) -> Self {
        // Seed, growth, and branch rolls are node-local and independent of graph expansion order.
        let mut rng = node_rng(graph, node);
        let parent_sides = ParentSides {
            graph: graph.parents(node).map(|(side, _)| side).collect(),
            tree: incoming_tree_parent_sides(graph, node),
        };
        let incoming = incoming_signals(graph, node);
        let (tree, parent_signal) =
            MegatreeNode::new(graph, node, kind, enviro, parent_sides, &incoming, &mut rng);
        let parent_signal_key = parent_signal.map(CarrierSignal::key);
        let mut diagonal = forward_carriers(&incoming, parent_signal_key, parent_sides.graph);
        let new_branches =
            generate_diagonal_branches(node, tree, kind, enviro, parent_sides, &mut rng);
        diagonal.outgoing.extend(new_branches.outgoing);
        diagonal.segments.extend(new_branches.segments);
        diagonal.has_children = new_branches.has_children;

        let diagonal = (parent_signal.is_some()
            || !diagonal.outgoing.is_empty()
            || !diagonal.segments.is_empty()
            || diagonal.has_children)
            .then(|| Box::new(diagonal));
        Self { tree, diagonal }
    }

    fn outgoing(&self) -> &[(Side, CarrierSignal)] {
        self.diagonal
            .as_deref()
            .map_or(&[], |diagonal| diagonal.outgoing.as_slice())
    }
}

/// Renders incoming carrier segments and relays unfinished signals through graph-child sides.
///
/// A completed signal is rendered only when selected as this node's parent; relaying never creates
/// a tree node or consumes growth.
fn forward_carriers(
    incoming: &[CarrierSignal],
    parent_signal: Option<SignalKey>,
    graph_parent_sides: SideSet,
) -> CarrierState {
    let mut state = CarrierState {
        outgoing: Vec::new(),
        segments: incoming
            .iter()
            .filter(|signal| !signal.is_complete() || parent_signal == Some(signal.key()))
            .map(|signal| signal.segment())
            .collect(),
        has_children: false,
    };

    for signal in incoming
        .iter()
        .copied()
        .filter(|signal| !signal.is_complete())
    {
        for side in Side::iter().filter(|side| signal.remaining_sides.contains(*side)) {
            if !graph_parent_sides.contains(side) {
                push_unique_outgoing(&mut state.outgoing, side, signal.cross(side));
            }
        }
    }
    state
}

fn generate_diagonal_branches(
    node: NodeId,
    tree: Option<MegatreeNode>,
    kind: NodeStateKind,
    enviro: EnviroFactors,
    parent_sides: ParentSides,
    rng: &mut Pcg64Mcg,
) -> CarrierState {
    let mut state = CarrierState {
        outgoing: Vec::new(),
        segments: Vec::new(),
        has_children: false,
    };
    if let Some(tree_node) = tree
        && matches!(kind, NodeStateKind::Sky | NodeStateKind::DeepSky)
        && tree_node.remaining_growth >= 0.0
    {
        for target_sides in diagonal_destinations(parent_sides.tree) {
            let routes = Side::iter()
                .filter(|side| target_sides.contains(*side) && !parent_sides.graph.contains(*side))
                .collect::<SideSet>();
            // Keep one roll per geometric destination, even if graph routing blocks it.
            let propagates = rng.random::<f32>() < branch_probability(enviro);
            if routes.is_empty() || !propagates {
                continue;
            }

            state.has_children = true;
            let signal = CarrierSignal {
                source: node,
                target_sides,
                remaining_sides: target_sides,
            };
            state.segments.push(signal.segment());
            for side in Side::iter().filter(|side| routes.contains(*side)) {
                push_unique_outgoing(&mut state.outgoing, side, signal.cross(side));
            }
        }
    }
    state
}

fn node_rng(graph: &Graph, node: NodeId) -> Pcg64Mcg {
    let spice = graph.hash_of(node) as u64;
    Pcg64Mcg::seed_from_u64(super::hash(spice, SEED))
}

#[derive(Clone, Copy)]
struct CarrierSignal {
    /// The source keeps branches distinct and retains the growth budget used at completion.
    ///
    /// Carrier routing only follows graph children, so the source state is finalized before
    /// every carrier and remains available and unchanged when the destination is generated.
    source: NodeId,
    /// Canonical crossing set and the subset still to traverse.
    target_sides: SideSet,
    remaining_sides: SideSet,
}

/// Graph parents determine which routes can be emitted; tree parents determine the growth cone.
#[derive(Clone, Copy)]
struct ParentSides {
    graph: SideSet,
    tree: SideSet,
}

impl CarrierSignal {
    fn crossed_sides(self) -> SideSet {
        Side::iter()
            .filter(|side| {
                self.target_sides.contains(*side) && !self.remaining_sides.contains(*side)
            })
            .collect()
    }

    fn key(self) -> SignalKey {
        (
            self.source,
            side_set_key(self.target_sides),
            side_set_key(self.remaining_sides),
        )
    }

    fn segment(self) -> LineSegment {
        let start = transform_sides(self.crossed_sides()) * MPoint::origin();
        let end = transform_sides(self.remaining_sides) * MPoint::origin();
        LineSegment::new(start, end)
    }

    fn cross(mut self, side: Side) -> Self {
        debug_assert!(self.remaining_sides.contains(side));
        self.remaining_sides.remove(side);
        self
    }

    fn is_complete(self) -> bool {
        self.remaining_sides.is_empty()
    }
}

fn side_set_key(sides: SideSet) -> u16 {
    Side::iter()
        .filter(|side| sides.contains(*side))
        .fold(0, |key, side| key | (1 << side as u8))
}

fn transform_sides(sides: SideSet) -> crate::math::MIsometry<f32> {
    Side::iter()
        .filter(|side| sides.contains(*side))
        .fold(crate::math::MIsometry::identity(), |transform, side| {
            side.reflection() * transform
        })
}

fn push_unique_signal(signals: &mut Vec<CarrierSignal>, signal: CarrierSignal) {
    if !signals
        .iter()
        .any(|existing| existing.key() == signal.key())
    {
        signals.push(signal);
    }
}

fn push_unique_outgoing(
    outgoing: &mut Vec<(Side, CarrierSignal)>,
    side: Side,
    signal: CarrierSignal,
) {
    if !outgoing
        .iter()
        .any(|(existing_side, existing)| *existing_side == side && existing.key() == signal.key())
    {
        outgoing.push((side, signal));
    }
}

fn incoming_signals(graph: &Graph, node: NodeId) -> Vec<CarrierSignal> {
    let mut signals = Vec::new();
    for (side, parent) in graph.parents(node) {
        for (outgoing_side, signal) in graph.node_state(parent).megatree.outgoing() {
            if *outgoing_side == side {
                push_unique_signal(&mut signals, *signal);
            }
        }
    }
    signals
}

fn incoming_tree_parent_sides(graph: &Graph, node: NodeId) -> SideSet {
    graph
        .parents(node)
        .filter_map(|(side, parent)| {
            let megatree = &graph.node_state(parent).megatree;
            let has_face_branch = megatree
                .tree
                .is_some_and(|tree| tree.child_sides.contains(side));
            let has_diagonal_branch = megatree
                .outgoing()
                .iter()
                .any(|(outgoing_side, _)| *outgoing_side == side);
            (has_face_branch || has_diagonal_branch).then_some(side)
        })
        .collect()
}

fn propagation_sides(parent_sides: SideSet) -> SideSet {
    eligible_destinations(parent_sides, 1, 1)
        .flat_map(|set| Side::iter().filter(move |side| set.contains(*side)))
        .collect()
}

fn diagonal_destinations(parent_sides: SideSet) -> impl Iterator<Item = SideSet> {
    eligible_destinations(parent_sides, 2, 3)
}

fn eligible_destinations(
    parent_sides: SideSet,
    min_crossings: usize,
    max_crossings: usize,
) -> impl Iterator<Item = SideSet> {
    let forbidden_sides = bounding_plane_sides(parent_sides);
    Side::iter()
        .map(|side| SideSet::from_iter([side]))
        .chain(EDGE_CROSSING_SETS.iter().copied())
        .chain(VERTEX_CROSSING_SETS.iter().copied())
        .filter(move |set| {
            (min_crossings..=max_crossings).contains(&set.len())
                && set.intersection(forbidden_sides).is_empty()
        })
}

static EDGE_CROSSING_SETS: LazyLock<Vec<SideSet>> = LazyLock::new(|| {
    Side::iter()
        .flat_map(|side| {
            Side::iter()
                .filter(move |other| *other > side && side.adjacent_to(*other))
                .map(move |other| SideSet::from_iter([side, other]))
        })
        .collect()
});

static VERTEX_CROSSING_SETS: LazyLock<Vec<SideSet>> = LazyLock::new(|| {
    Vertex::iter()
        .map(Vertex::canonical_sides)
        .map(SideSet::from_iter)
        .collect()
});

fn bounding_plane_sides(parent_sides: SideSet) -> SideSet {
    match parent_sides.len() {
        1 => {
            let parent = Side::iter()
                .find(|side| parent_sides.contains(*side))
                .unwrap();
            Side::iter()
                .filter(|side| *side == parent || side.adjacent_to(parent))
                .collect()
        }
        2 => {
            let mut parents = Side::iter().filter(|side| parent_sides.contains(*side));
            let first = parents.next().expect("two parent sides are present");
            let second = parents.next().expect("two parent sides are present");
            parent_sides.union(
                Side::iter()
                    .filter(|side| side.adjacent_to(first) && side.adjacent_to(second))
                    .collect(),
            )
        }
        _ => parent_sides,
    }
}

fn graph_child_sides(parent_sides: SideSet) -> SideSet {
    Side::iter()
        .filter(|side| !parent_sides.contains(*side))
        .collect()
}

/// Megatree propagation information for one node.
///
/// `parent_side` identifies the incoming branch; `child_sides` records outgoing
/// propagation directions. Rendering uses both, while propagation uses children only.
#[derive(Clone, Copy)]
struct MegatreeNode {
    parent_side: Option<Side>,
    child_sides: SideSet,
    remaining_growth: f32,
}

impl MegatreeNode {
    fn new(
        graph: &Graph,
        node: NodeId,
        kind: NodeStateKind,
        enviro: EnviroFactors,
        parent_sides: ParentSides,
        incoming: &[CarrierSignal],
        rng: &mut Pcg64Mcg,
    ) -> (Option<Self>, Option<CarrierSignal>) {
        // `Graph::parents` returns every shallower neighbor and the shared side.
        // Parent states have already been finalized by `ensure_node_state`.
        let ParentSides {
            graph: graph_parent_sides,
            tree: tree_parent_sides,
        } = parent_sides;
        let face_parent = graph
            .parents(node)
            .filter_map(|(side, parent)| {
                graph.node_state(parent).megatree.tree.and_then(|state| {
                    state
                        .child_sides
                        .contains(side)
                        .then_some((side, state.remaining_growth))
                })
            })
            .min_by_key(|(side, _)| *side as u8);
        // Prefer the existing face path; stable keys resolve multiple incoming paths predictably.
        let diagonal_parent = incoming
            .iter()
            .copied()
            .filter(|signal| signal.is_complete())
            .min_by_key(|signal| signal.key());
        let parent = face_parent
            .map(|(side, growth)| (Some(side), growth, SideSet::from_iter([side]), None))
            .or_else(|| {
                diagonal_parent.map(|signal| {
                    (
                        None,
                        graph
                            .node_state(signal.source)
                            .megatree
                            .tree
                            .expect("diagonal signals originate from a tree node")
                            .remaining_growth,
                        signal.target_sides,
                        Some(signal),
                    )
                })
            });

        if let Some((parent_side, parent_growth, incoming_sides, parent_signal)) = parent {
            // Continuation path: inherit growth and select children from the parent side.
            let remaining_growth = if matches!(kind, NodeStateKind::Sky | NodeStateKind::DeepSky) {
                parent_growth - 1.0
                    + rng.sample(Normal::new(0.0, GROWTH_COST_STANDARD_DEVIATION).unwrap())
            } else {
                parent_growth
            };
            let mut child_sides = SideSet::EMPTY;
            if remaining_growth >= 0.0 {
                if kind != NodeStateKind::DeepLand {
                    let eligible_sides = propagation_sides(tree_parent_sides)
                        .intersection(graph_child_sides(graph_parent_sides));
                    for side in Side::iter().filter(|side| eligible_sides.contains(*side)) {
                        if rng.random::<f32>() < branch_probability(enviro) {
                            child_sides.insert(side);
                        }
                    }
                } else {
                    let trunk_side = parent_side.unwrap_or_else(|| {
                        Side::iter()
                            .find(|side| incoming_sides.contains(*side))
                            .expect("a diagonal tree node has incoming crossing sides")
                    });
                    child_sides.insert(trunk_side.opposite());
                }
            }

            (
                Some(Self {
                    parent_side,
                    child_sides,
                    remaining_growth,
                }),
                parent_signal,
            )
        } else {
            // Seed path: new trees start on Land and grow along the ground-plane axis.
            if kind == NodeStateKind::Land
                && rng.random::<f32>() < tree_generation_probability(enviro)
            {
                (
                    Some(Self {
                        parent_side: None,
                        child_sides: SideSet::from_iter([Side::A, Side::A.opposite()]),
                        remaining_growth: INITIAL_REMAINING_GROWTH,
                    }),
                    None,
                )
            } else {
                (None, None)
            }
        }
    }
}

/// Megatree geometry needed to render one chunk.
pub struct MegatreeChunk {
    dimension: u8,
    chunk: Vertex,
    branches: Vec<LineSegment>,
    has_leaves: bool,
}

impl MegatreeChunk {
    pub fn new(state: &MegatreeState, dimension: u8, chunk: Vertex) -> Option<Self> {
        let mut branches = state
            .diagonal
            .as_ref()
            .map_or_else(Vec::new, |diagonal| diagonal.segments.clone());
        let has_diagonal_children = state
            .diagonal
            .as_ref()
            .is_some_and(|diagonal| diagonal.has_children);
        let has_leaves = state
            .tree
            .is_some_and(|tree| tree.child_sides.is_empty() && !has_diagonal_children);
        if let Some(tree) = state.tree {
            branches.extend(Self::from_tree(tree, dimension, chunk).branches);
        }
        (!branches.is_empty() || has_leaves).then_some(Self {
            dimension,
            chunk,
            branches,
            has_leaves,
        })
    }

    fn from_tree(megatree: MegatreeNode, dimension: u8, chunk: Vertex) -> Self {
        let branches = megatree
            .parent_side
            .into_iter()
            .chain(Side::iter().filter(move |&side| megatree.child_sides.contains(side)))
            .map(|side| {
                let center = MPoint::origin();
                let neighbor_center = side.reflection() * center;
                let edge_center = center.midpoint(&neighbor_center);
                LineSegment::new(center, edge_center)
            })
            .collect();
        Self {
            dimension,
            chunk,
            branches,
            has_leaves: megatree.child_sides.is_empty(),
        }
    }

    pub fn generate(&self, voxels: &mut VoxelData) {
        if self.branches.is_empty() && !self.has_leaves {
            return;
        }

        for (x, y, z) in VoxelCoords::new(self.dimension) {
            let coords = na::Vector3::new(x, y, z);
            let voxel_index = index(self.dimension, coords);
            if voxels.get(voxel_index) != Material::Void {
                continue;
            }
            let center = voxel_center(self.dimension, coords);
            let point =
                MVector::from(self.chunk.chunk_to_node() * center.push(1.0)).normalized_point();

            let material = if self
                .branches
                .iter()
                .any(|branch| branch.distance_to(&point) <= BRANCH_RADIUS)
            {
                Material::Wood
            } else if self.has_leaves && point.distance(&MPoint::origin()) <= LEAVES_RADIUS {
                Material::Leaves
            } else {
                continue;
            };
            voxels.data_mut(self.dimension)[voxel_index] = material;
        }
    }
}

fn tree_generation_probability(enviro: EnviroFactors) -> f32 {
    (enviro.rainfall * 0.025).clamp(0.0, 1.0)
}

fn branch_probability(enviro: EnviroFactors) -> f32 {
    // Use a linear ramp with lower and upper temperature limits.
    const BRANCH_TEMPERATURE_MIN: f32 = -10.0;
    const BRANCH_TEMPERATURE_MAX: f32 = 10.0;
    ((enviro.temperature - BRANCH_TEMPERATURE_MIN)
        / (BRANCH_TEMPERATURE_MAX - BRANCH_TEMPERATURE_MIN))
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::WorldgenConfig;
    use std::collections::HashSet;

    fn climate(rainfall: f32, temperature: f32) -> EnviroFactors {
        EnviroFactors {
            max_elevation: 0.0,
            rainfall,
            temperature,
            blockiness: 0.0,
        }
    }

    fn inherited_tree(kind: NodeStateKind) -> MegatreeNode {
        let mut graph = Graph::new(1);
        graph.ensure_node_state(NodeId::ROOT, &WorldgenConfig::default());
        let mut parent = MegatreeNode {
            parent_side: None,
            child_sides: SideSet::EMPTY,
            remaining_growth: 2.5,
        };
        parent.child_sides.insert(Side::B);
        graph[NodeId::ROOT].state.as_mut().unwrap().megatree.tree = Some(parent);
        let child = graph.ensure_neighbor(NodeId::ROOT, Side::B);
        let mut rng = node_rng(&graph, child);
        MegatreeNode::new(
            &graph,
            child,
            kind,
            climate(0.0, 10.0),
            ParentSides {
                graph: SideSet::from_iter([Side::B]),
                tree: SideSet::from_iter([Side::B]),
            },
            &[],
            &mut rng,
        )
        .0
        .unwrap()
    }

    #[test]
    fn sky_branches_avoid_parent_and_adjacent_faces() {
        // Sky branches choose every eligible side independently when the branch probability is 1.
        let tree = inherited_tree(NodeStateKind::DeepSky);
        let child_sides = tree.child_sides;
        assert_eq!(child_sides.len(), 6);
        assert!(!child_sides.contains(Side::B));
        assert!(
            Side::iter()
                .filter(|s| s.adjacent_to(Side::B))
                .all(|s| !child_sides.contains(s))
        );
    }

    #[test]
    fn deep_land_always_continues_as_trunk() {
        // DeepLand always continues the trunk through the face opposite its incoming parent.
        let tree = inherited_tree(NodeStateKind::DeepLand);
        assert_eq!(tree.child_sides, SideSet::from_iter([Side::B.opposite()]));
    }

    #[test]
    fn land_seed_uses_side_a_as_ground_axis() {
        // A newly seeded Land tree begins along the ground-plane axis and its opposite face.
        let graph = Graph::new(1);
        let mut rng = node_rng(&graph, NodeId::ROOT);
        let (tree, _) = MegatreeNode::new(
            &graph,
            NodeId::ROOT,
            NodeStateKind::Land,
            climate(100.0, 10.0),
            ParentSides {
                graph: SideSet::EMPTY,
                tree: SideSet::EMPTY,
            },
            &[],
            &mut rng,
        );
        let tree = tree.unwrap();
        assert_eq!(
            tree.child_sides,
            SideSet::from_iter([Side::A, Side::A.opposite()])
        );
        assert_eq!(tree.parent_side, None);
    }

    #[test]
    fn diagonal_destinations_match_the_geometric_rule_for_every_parent_feature() {
        let features = all_crossing_features();
        assert_eq!(features.len(), 62);

        for parent_sides in &features {
            let expected = match parent_sides.len() {
                1 => [6, 10, 5],
                2 => [8, 15, 8],
                3 => [9, 18, 10],
                _ => unreachable!(),
            };
            let mut actual = [0; 3];
            for destination in eligible_destinations(*parent_sides, 1, 3) {
                actual[destination.len() - 1] += 1;
            }
            assert_eq!(actual, expected, "for parents {parent_sides:?}");

            let forbidden = bounding_plane_sides(*parent_sides);
            for destination in &features {
                let expected_to_cross =
                    !eligible_destinations(*parent_sides, 1, 3).any(|set| set == *destination);
                let crosses_by_side_set = !destination.intersection(forbidden).is_empty();
                assert_eq!(
                    expected_to_cross, crosses_by_side_set,
                    "for parents {parent_sides:?} and destination {destination:?}"
                );
            }
        }
    }

    #[test]
    fn propagation_rule_is_equivariant_under_every_dodecahedral_symmetry() {
        // Every face, edge, and vertex parent configuration must preserve its full exit set.
        let symmetries = side_adjacency_automorphisms();
        assert_eq!(symmetries.len(), 120);

        for parents in all_crossing_features() {
            let outgoing = eligible_destinations(parents, 1, 3).collect::<Vec<_>>();
            for symmetry in &symmetries {
                let transformed_parents = transform_side_set(parents, symmetry);
                let transformed_outgoing =
                    eligible_destinations(transformed_parents, 1, 3).collect::<HashSet<_>>();
                let expected_outgoing = outgoing
                    .iter()
                    .map(|set| transform_side_set(*set, symmetry))
                    .collect::<HashSet<_>>();
                assert_eq!(transformed_outgoing, expected_outgoing);
            }
        }
    }

    #[test]
    fn sky_nodes_emit_one_canonical_signal_per_valid_diagonal_route() {
        // One tree parent permits 10 edge and 5 vertex destinations; each route is sent once.
        let cfg = WorldgenConfig::default();
        let mut graph = Graph::new(1);
        graph.ensure_node_state(NodeId::ROOT, &cfg);
        graph[NodeId::ROOT].state.as_mut().unwrap().megatree.tree = Some(MegatreeNode {
            parent_side: None,
            child_sides: SideSet::from_iter([Side::A]),
            remaining_growth: INITIAL_REMAINING_GROWTH,
        });
        let node = graph.ensure_neighbor(NodeId::ROOT, Side::A);
        graph.ensure_node_state(node, &cfg);

        let state = MegatreeState::new(&graph, node, NodeStateKind::Sky, climate(0.0, 10.0));
        let tree = state.tree.unwrap();
        let diagonal = state.diagonal.unwrap();

        assert_eq!(diagonal.segments.len(), 15);
        assert_eq!(diagonal.outgoing.len(), 35);
        assert!(diagonal.has_children);
        assert!(diagonal.outgoing.iter().all(|(_, signal)| {
            graph
                .node_state(signal.source)
                .megatree
                .tree
                .unwrap()
                .remaining_growth
                == tree.remaining_growth
        }));
    }

    #[test]
    fn edge_and_vertex_routes_merge_without_spending_growth_on_carriers() {
        // Crossing orders converge to canonical signals at the (1,2,1) and (1,3,3,1) layers.
        for crossing_count in [2, 3] {
            let parents = SideSet::from_iter([Side::A]);
            let target_sides = diagonal_destinations(parents)
                .find(|target| target.len() == crossing_count)
                .unwrap();
            let (graph_a, target_a, layers_a) = graph_with_carriers(target_sides, false);
            let (graph_b, target_b, layers_b) = graph_with_carriers(target_sides, true);
            assert_eq!(target_a, target_b);
            assert_eq!(layers_a, layers_b);
            assert_eq!(layers_a[0].len(), crossing_count);
            if crossing_count == 3 {
                assert_eq!(layers_a[1].len(), 3);
            }

            for (layer_index, layer) in layers_a.iter().enumerate() {
                for carrier in layer {
                    let state = &graph_a.node_state(*carrier).megatree;
                    assert!(state.tree.is_none());
                    let diagonal = state.diagonal.as_ref().unwrap();
                    assert_eq!(diagonal.segments.len(), 1);
                    assert!(diagonal.outgoing.iter().all(|(_, signal)| {
                        graph_a
                            .node_state(signal.source)
                            .megatree
                            .tree
                            .unwrap()
                            .remaining_growth
                            == INITIAL_REMAINING_GROWTH
                    }));
                    if crossing_count == 3 && layer_index == 0 {
                        assert_eq!(diagonal.outgoing.len(), 2);
                    }
                    if crossing_count == 3 && layer_index == 1 {
                        assert_eq!(diagonal.outgoing.len(), 1);
                    }

                    let chunk = MegatreeChunk::new(state, 12, Vertex::A).unwrap();
                    assert!(!chunk.has_leaves);
                    assert_eq!(chunk.branches.len(), 1);
                }
            }

            let state_a = MegatreeState::new(
                &graph_a,
                target_a,
                NodeStateKind::DeepSky,
                climate(0.0, -10.0),
            );
            let state_b = MegatreeState::new(
                &graph_b,
                target_b,
                NodeStateKind::DeepSky,
                climate(0.0, -10.0),
            );
            let tree_a = state_a.tree.unwrap();
            let tree_b = state_b.tree.unwrap();
            let mut expected_rng = node_rng(&graph_a, target_a);
            let expected_growth = INITIAL_REMAINING_GROWTH - 1.0
                + expected_rng.sample(Normal::new(0.0, GROWTH_COST_STANDARD_DEVIATION).unwrap());

            assert_eq!(tree_a.parent_side, None);
            assert_eq!(tree_a.remaining_growth, expected_growth);
            assert_eq!(tree_a.remaining_growth, tree_b.remaining_growth);
            assert_eq!(tree_a.child_sides, tree_b.child_sides);
            assert_eq!(state_a.diagonal.as_ref().unwrap().segments.len(), 1);
            assert_eq!(state_b.diagonal.as_ref().unwrap().segments.len(), 1);
        }
    }

    #[test]
    fn face_parent_wins_a_diagonal_convergence_deterministically() {
        // A direct face path keeps its existing growth inheritance when a diagonal signal also arrives.
        let mut outcomes = Vec::new();
        for order in [[Side::A, Side::B], [Side::B, Side::A]] {
            let cfg = WorldgenConfig::default();
            let mut graph = Graph::new(1);
            graph.ensure_node_state(NodeId::ROOT, &cfg);
            let first = graph.ensure_neighbor(NodeId::ROOT, order[0]);
            let second = graph.ensure_neighbor(NodeId::ROOT, order[1]);
            let target = graph.ensure_neighbor(first, order[1]);
            assert_eq!(target, graph.ensure_neighbor(second, order[0]));
            for (_, parent) in graph.parents(target) {
                graph.ensure_node_state(parent, &cfg);
            }

            for (side, parent) in graph.parents(target) {
                let megatree = &mut graph[parent].state.as_mut().unwrap().megatree;
                megatree.tree = Some(MegatreeNode {
                    parent_side: None,
                    child_sides: SideSet::from_iter([side]),
                    remaining_growth: if side == Side::A { 2.5 } else { 7.5 },
                });
            }
            let (diagonal_side, diagonal_parent) = graph
                .parents(target)
                .find(|(side, _)| *side == Side::A)
                .unwrap();
            graph[diagonal_parent]
                .state
                .as_mut()
                .unwrap()
                .megatree
                .diagonal = Some(Box::new(CarrierState {
                outgoing: vec![(
                    diagonal_side,
                    CarrierSignal {
                        source: NodeId::ROOT,
                        target_sides: SideSet::from_iter([Side::A, Side::B]),
                        remaining_sides: SideSet::EMPTY,
                    },
                )],
                segments: Vec::new(),
                has_children: false,
            }));

            let state =
                MegatreeState::new(&graph, target, NodeStateKind::DeepSky, climate(0.0, -10.0));
            let tree = state.tree.unwrap();
            let mut rng = node_rng(&graph, target);
            let expected_growth =
                2.5 - 1.0 + rng.sample(Normal::new(0.0, GROWTH_COST_STANDARD_DEVIATION).unwrap());
            assert_eq!(tree.parent_side, Some(Side::A));
            assert_eq!(tree.remaining_growth, expected_growth);
            assert!(state.diagonal.is_none());
            outcomes.push((tree.parent_side, tree.remaining_growth));
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }

    #[test]
    fn negative_growth_stops_a_completed_diagonal_tree() {
        // A completed carrier charges growth once, then a negative budget produces a terminal node.
        let mut graph = Graph::new(1);
        graph.ensure_node_state(NodeId::ROOT, &WorldgenConfig::default());
        graph[NodeId::ROOT].state.as_mut().unwrap().megatree.tree = Some(MegatreeNode {
            parent_side: None,
            child_sides: SideSet::EMPTY,
            remaining_growth: -10.0,
        });
        let signal = CarrierSignal {
            source: NodeId::ROOT,
            target_sides: SideSet::from_iter([Side::A, Side::B]),
            remaining_sides: SideSet::EMPTY,
        };
        let mut rng = node_rng(&graph, NodeId::ROOT);
        let (tree, parent_signal) = MegatreeNode::new(
            &graph,
            NodeId::ROOT,
            NodeStateKind::DeepSky,
            climate(0.0, 10.0),
            ParentSides {
                graph: SideSet::EMPTY,
                tree: signal.target_sides,
            },
            &[signal],
            &mut rng,
        );
        let tree = tree.unwrap();
        assert!(tree.remaining_growth < 0.0);
        assert!(tree.child_sides.is_empty());
        assert_eq!(parent_signal.map(CarrierSignal::key), Some(signal.key()));
    }

    #[test]
    fn diagonal_segments_connect_the_centers_on_each_side_of_a_carrier() {
        // Remaining crossing sides locate the target endpoint; the crossed complement locates the source.
        let targets = [
            SideSet::from_iter([Side::A, Side::B]),
            SideSet::from_iter(Vertex::A.canonical_sides()),
        ];
        for target_sides in targets {
            let crossed = Side::iter()
                .filter(|side| target_sides.contains(*side))
                .take(target_sides.len() - 1)
                .collect::<SideSet>();
            let remaining = target_sides.intersection(
                Side::iter()
                    .filter(|side| !crossed.contains(*side))
                    .collect(),
            );
            let signal = CarrierSignal {
                source: NodeId::ROOT,
                target_sides,
                remaining_sides: remaining,
            };
            let start = transform_sides(crossed) * MPoint::origin();
            let end = transform_sides(remaining) * MPoint::origin();
            let segment = signal.segment();
            assert!(segment.distance_to(&start) < 1e-5);
            assert!(segment.distance_to(&end) < 1e-5);
        }
    }

    fn all_crossing_features() -> Vec<SideSet> {
        Side::iter()
            .map(|side| SideSet::from_iter([side]))
            .chain(EDGE_CROSSING_SETS.iter().copied())
            .chain(VERTEX_CROSSING_SETS.iter().copied())
            .collect()
    }

    fn transform_side_set(sides: SideSet, symmetry: &[Side; Side::VALUES.len()]) -> SideSet {
        Side::iter()
            .filter(|side| sides.contains(*side))
            .map(|side| symmetry[side as usize])
            .collect()
    }

    fn side_adjacency_automorphisms() -> Vec<[Side; Side::VALUES.len()]> {
        fn extend(
            mapping: &mut [Option<Side>; Side::VALUES.len()],
            used: u16,
            result: &mut Vec<[Side; Side::VALUES.len()]>,
        ) {
            let next = Side::iter()
                .filter(|side| mapping[*side as usize].is_none())
                .max_by_key(|side| {
                    Side::iter()
                        .filter(|other| {
                            mapping[*other as usize].is_some() && side.adjacent_to(*other)
                        })
                        .count()
                });
            let Some(source) = next else {
                result.push(mapping.map(Option::unwrap));
                return;
            };

            for candidate in Side::iter().filter(|side| used & (1 << (*side as u8)) == 0) {
                let preserves_adjacency = Side::iter()
                    .filter_map(|other| mapping[other as usize].map(|mapped| (other, mapped)))
                    .all(|(other, mapped)| {
                        source.adjacent_to(other) == candidate.adjacent_to(mapped)
                    });
                if preserves_adjacency {
                    mapping[source as usize] = Some(candidate);
                    extend(mapping, used | (1 << (candidate as u8)), result);
                    mapping[source as usize] = None;
                }
            }
        }

        let mut result = Vec::new();
        extend(&mut [None; Side::VALUES.len()], 0, &mut result);
        result
    }

    fn graph_with_carriers(
        target_sides: SideSet,
        reverse_order: bool,
    ) -> (Graph, NodeId, Vec<Vec<NodeId>>) {
        let cfg = WorldgenConfig::default();
        let mut graph = Graph::new(1);
        graph.ensure_node_state(NodeId::ROOT, &cfg);
        let source = graph.ensure_neighbor(NodeId::ROOT, Side::A);

        let mut crossing_orders = permutations(target_sides);
        if reverse_order {
            crossing_orders.reverse();
        }
        let mut paths = Vec::new();
        for permutation in crossing_orders {
            let mut at = source;
            let mut path = vec![source];
            for side in permutation {
                at = graph.ensure_neighbor(at, side);
                path.push(at);
            }
            paths.push(path);
        }
        let target = paths[0].last().copied().unwrap();
        assert!(paths.iter().all(|path| path.last() == Some(&target)));
        graph.ensure_node_state(source, &cfg);

        // Seed one canonical carrier route directly so face propagation cannot mask its convergence.
        graph[source].state.as_mut().unwrap().megatree = MegatreeState {
            tree: Some(MegatreeNode {
                parent_side: Some(Side::A.opposite()),
                child_sides: SideSet::EMPTY,
                remaining_growth: INITIAL_REMAINING_GROWTH,
            }),
            diagonal: Some(Box::new(CarrierState {
                outgoing: Side::iter()
                    .filter(|side| target_sides.contains(*side))
                    .map(|side| {
                        (
                            side,
                            CarrierSignal {
                                source,
                                target_sides,
                                remaining_sides: target_sides,
                            }
                            .cross(side),
                        )
                    })
                    .collect(),
                segments: Vec::new(),
                has_children: true,
            })),
        };

        let mut intermediate = paths
            .iter()
            .flat_map(|path| path.iter().copied().skip(1).take(target_sides.len() - 1))
            .collect::<Vec<_>>();
        intermediate.sort_by_key(|node| (graph.depth(*node), *node));
        intermediate.dedup();
        // Finalize carrier states in graph-depth order after varying insertion order above.
        for node in intermediate {
            graph.ensure_node_state(node, &cfg);
            let state =
                MegatreeState::new(&graph, node, NodeStateKind::DeepSky, climate(0.0, -10.0));
            graph[node].state.as_mut().unwrap().megatree = state;
        }

        let mut layers = vec![Vec::new(); target_sides.len() - 1];
        for path in &paths {
            for (index, node) in path
                .iter()
                .copied()
                .skip(1)
                .take(target_sides.len() - 1)
                .enumerate()
            {
                layers[index].push(node);
            }
        }
        for layer in &mut layers {
            layer.sort_unstable();
            layer.dedup();
        }
        (graph, target, layers)
    }

    fn permutations(sides: SideSet) -> Vec<Vec<Side>> {
        fn append(remaining: &mut Vec<Side>, prefix: &mut Vec<Side>, result: &mut Vec<Vec<Side>>) {
            if remaining.is_empty() {
                result.push(prefix.clone());
                return;
            }
            for index in 0..remaining.len() {
                let side = remaining.remove(index);
                prefix.push(side);
                append(remaining, prefix, result);
                prefix.pop();
                remaining.insert(index, side);
            }
        }

        let mut remaining = Side::iter().filter(|side| sides.contains(*side)).collect();
        let mut result = Vec::new();
        append(&mut remaining, &mut Vec::new(), &mut result);
        result
    }

    #[test]
    fn rendering_fills_void_but_preserves_terrain() {
        // Megatree voxels preserve solid terrain and fill only void portions of mixed chunks.
        let tree = MegatreeNode {
            parent_side: Some(Side::B),
            child_sides: SideSet::EMPTY,
            remaining_growth: 2.5,
        };
        let chunk = MegatreeChunk::from_tree(tree, 12, Vertex::A);
        let mut reference = VoxelData::Solid(Material::Void);
        chunk.generate(&mut reference);
        let size = (12usize + 2).pow(3);
        let wood = (0..size)
            .find(|&i| reference.get(i) == Material::Wood)
            .unwrap();
        let leaves = (0..size)
            .find(|&i| reference.get(i) == Material::Leaves)
            .unwrap();

        let mut solid = VoxelData::Solid(Material::Dirt);
        chunk.generate(&mut solid);
        assert!(matches!(solid, VoxelData::Solid(Material::Dirt)));

        let mut mixed = VoxelData::Solid(Material::Void);
        mixed.data_mut(12)[wood] = Material::Dirt;
        mixed.data_mut(12)[leaves] = Material::Dirt;
        chunk.generate(&mut mixed);
        assert_eq!(mixed.get(wood), Material::Dirt);
        assert_eq!(mixed.get(leaves), Material::Dirt);
        assert!((0..size).any(|i| mixed.get(i) == Material::Wood));
        assert!((0..size).any(|i| mixed.get(i) == Material::Leaves));
    }
}
