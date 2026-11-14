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
///
/// Carrier signals only cross sides that lead to graph children. Therefore every
/// carrier that can contribute to a node is one of its graph parents, whose state
/// `ensure_node_state` finalizes before this state is constructed.
pub struct MegatreeState {
    tree: Option<MegatreeNode>,
    diagonal: Option<Box<DiagonalState>>,
}

#[derive(Default)]
struct DiagonalState {
    outgoing: Vec<(Side, CarrierSignal)>,
    segments: Vec<LineSegment>,
    has_children: bool,
}

impl MegatreeState {
    pub fn new(graph: &Graph, node: NodeId, kind: NodeStateKind, enviro: EnviroFactors) -> Self {
        let graph_parent_sides = graph
            .parents(node)
            .map(|(side, _)| side)
            .collect::<SideSet>();
        let tree_parent_sides = incoming_tree_parent_sides(graph, node);
        let (tree, parent_signal) = MegatreeNode::new(
            graph,
            node,
            kind,
            enviro,
            tree_parent_sides,
            graph_parent_sides,
        );
        let mut incoming_signals = Vec::new();
        for (side, parent) in graph.parents(node) {
            for (outgoing_side, signal) in graph.node_state(parent).megatree.outgoing() {
                if *outgoing_side == side {
                    incoming_signals.push(*signal);
                }
            }
        }
        incoming_signals.sort_by_key(|signal| signal.key());
        incoming_signals.dedup_by_key(|signal| signal.key());
        let mut segments = incoming_signals
            .iter()
            .copied()
            .filter(|signal| {
                !signal.is_complete() || parent_signal == Some((signal.source, signal.target_sides))
            })
            .map(CarrierSignal::segment)
            .collect::<Vec<_>>();
        let carriers = incoming_signals
            .into_iter()
            .filter(|signal| !signal.is_complete())
            .collect::<Vec<_>>();

        let mut outgoing = Vec::new();
        let mut has_diagonal_children = false;
        for signal in carriers.iter().copied() {
            for side in Side::iter().filter(|side| signal.remaining_sides().contains(*side)) {
                // A carrier step must go to a graph child so its state has not
                // already been finalized. Other crossing orders may still
                // reach the same completed destination.
                if graph_parent_sides.contains(side) {
                    continue;
                }
                let mut next = signal;
                next.crossed_sides.insert(side);
                outgoing.push((side, next));
            }
        }

        if let Some(tree_node) = tree
            && matches!(kind, NodeStateKind::Sky | NodeStateKind::DeepSky)
            && tree_node.remaining_growth >= 0.0
        {
            let spice = graph.hash_of(node) as u64;
            let mut rng = Pcg64Mcg::seed_from_u64(super::hash(spice, DIAGONAL_SEED));
            for target_sides in edge_propagation_sets(tree_parent_sides) {
                if rng.random::<f32>() < branch_probability(enviro) {
                    has_diagonal_children = true;
                    let signal = CarrierSignal {
                        source: node,
                        target_sides,
                        crossed_sides: SideSet::EMPTY,
                        remaining_growth: tree_node.remaining_growth,
                    };
                    segments.push(signal.segment());
                    for side in Side::iter().filter(|side| target_sides.contains(*side)) {
                        let mut next = signal;
                        next.crossed_sides.insert(side);
                        outgoing.push((side, next));
                    }
                }
            }
            for target_sides in vertex_propagation_sets(tree_parent_sides) {
                if rng.random::<f32>() < branch_probability(enviro) {
                    has_diagonal_children = true;
                    let signal = CarrierSignal {
                        source: node,
                        target_sides,
                        crossed_sides: SideSet::EMPTY,
                        remaining_growth: tree_node.remaining_growth,
                    };
                    segments.push(signal.segment());
                    for side in Side::iter().filter(|side| target_sides.contains(*side)) {
                        let mut next = signal;
                        next.crossed_sides.insert(side);
                        outgoing.push((side, next));
                    }
                }
            }
        }

        outgoing.sort_by_key(|(side, signal)| (*side, signal.key()));
        outgoing.dedup_by_key(|(side, signal)| (*side, signal.key()));

        let diagonal = (parent_signal.is_some()
            || !outgoing.is_empty()
            || !segments.is_empty()
            || has_diagonal_children)
            .then(|| {
                Box::new(DiagonalState {
                    outgoing,
                    segments,
                    has_children: has_diagonal_children,
                })
            });

        Self { tree, diagonal }
    }

    fn outgoing(&self) -> &[(Side, CarrierSignal)] {
        self.diagonal
            .as_deref()
            .map_or(&[], |diagonal| diagonal.outgoing.as_slice())
    }
}

const DIAGONAL_SEED: u64 = 10655026474825314811;

#[derive(Clone, Copy)]
struct CarrierSignal {
    source: NodeId,
    target_sides: SideSet,
    crossed_sides: SideSet,
    remaining_growth: f32,
}

impl CarrierSignal {
    fn remaining_sides(self) -> SideSet {
        Side::iter()
            .filter(|side| self.target_sides.contains(*side) && !self.crossed_sides.contains(*side))
            .collect()
    }

    fn key(self) -> (NodeId, u16, u16) {
        (
            self.source,
            side_set_key(self.target_sides),
            side_set_key(self.crossed_sides),
        )
    }

    fn segment(self) -> LineSegment {
        let start = transform_sides(self.crossed_sides) * MPoint::origin();
        let end = transform_sides(self.remaining_sides()) * MPoint::origin();
        LineSegment::new(start, end)
    }

    fn is_complete(self) -> bool {
        self.remaining_sides().is_empty()
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

static EDGE_CROSSING_SETS: LazyLock<Vec<SideSet>> = LazyLock::new(|| {
    let sides = Side::iter().collect::<Vec<_>>();
    let mut result = Vec::new();
    for (index, &side) in sides.iter().enumerate() {
        for &other in &sides[index + 1..] {
            if side.adjacent_to(other) {
                result.push(SideSet::from_iter([side, other]));
            }
        }
    }
    result
});

static VERTEX_CROSSING_SETS: LazyLock<Vec<SideSet>> = LazyLock::new(|| {
    Vertex::iter()
        .map(Vertex::canonical_sides)
        .map(SideSet::from_iter)
        .collect()
});

fn edge_crossing_sets(eligible_sides: SideSet) -> impl Iterator<Item = SideSet> {
    EDGE_CROSSING_SETS
        .iter()
        .copied()
        .filter(move |set| side_set_key(*set) & side_set_key(eligible_sides) == side_set_key(*set))
}

fn vertex_crossing_sets(eligible_sides: SideSet) -> impl Iterator<Item = SideSet> {
    VERTEX_CROSSING_SETS
        .iter()
        .copied()
        .filter(move |set| side_set_key(*set) & side_set_key(eligible_sides) == side_set_key(*set))
}

fn edge_propagation_sets(parent_sides: SideSet) -> impl Iterator<Item = SideSet> {
    edge_crossing_sets(SideSet::ALL_SIDES)
        .filter(move |target| !destination_crosses_parent_plane(parent_sides, *target))
}

fn vertex_propagation_sets(parent_sides: SideSet) -> impl Iterator<Item = SideSet> {
    vertex_crossing_sets(SideSet::ALL_SIDES)
        .filter(move |target| !destination_crosses_parent_plane(parent_sides, *target))
}

/// Face planes that bound propagation for this incoming parent feature.
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
            let parents = Side::iter()
                .filter(|side| parent_sides.contains(*side))
                .collect::<Vec<_>>();
            let [first, second] = parents[..] else {
                unreachable!()
            };
            parent_sides.union(
                Side::iter()
                    .filter(|side| side.adjacent_to(first) && side.adjacent_to(second))
                    .collect(),
            )
        }
        3 => parent_sides,
        _ => parent_sides,
    }
}

/// Whether the geodesic from this node to the completed destination crosses
/// any bounding plane for its incoming parent feature.
fn destination_crosses_parent_plane(parent_sides: SideSet, target_sides: SideSet) -> bool {
    let target_center = Side::iter()
        .filter(|side| target_sides.contains(*side))
        .fold(
            crate::math::MIsometry::<f64>::identity(),
            |transform, side| side.reflection_f64() * transform,
        )
        * MPoint::origin();

    Side::iter()
        .filter(|side| bounding_plane_sides(parent_sides).contains(*side))
        .any(|plane_side| {
            // The source center lies on the negative side of each plane. A
            // target on or beyond a plane makes its geodesic intersect it.
            plane_side.normal_f64().as_ref().mip(&target_center) >= -1.0e-10
        })
}

fn propagation_sides(parent_sides: SideSet) -> SideSet {
    Side::iter()
        .filter(|side| !destination_crosses_parent_plane(parent_sides, SideSet::from_iter([*side])))
        .collect()
}

/// Geometric exits are bounded by incoming tree branches, not every graph
/// ancestor. Graph ancestry is handled separately when routing the signal.
fn diagonal_propagation_sides(tree_parent_sides: SideSet) -> SideSet {
    propagation_sides(tree_parent_sides)
}

/// A node's state is finalized after its graph parents, so new face branches
/// can only be sent toward graph children.
fn graph_child_sides(graph_parent_sides: SideSet) -> SideSet {
    let graph_child_sides = Side::iter()
        .filter(|side| !graph_parent_sides.contains(*side))
        .collect::<SideSet>();
    graph_child_sides
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

/// Megatree propagation information for one node.
///
/// `parent_side` identifies a face-connected incoming branch, while `child_sides`
/// stores face-connected exits.
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
        tree_parent_sides: SideSet,
        graph_parent_sides: SideSet,
    ) -> (Option<Self>, Option<(NodeId, SideSet)>) {
        let spice = graph.hash_of(node) as u64;
        let mut rng = Pcg64Mcg::seed_from_u64(super::hash(spice, SEED));
        // `Graph::parents` returns every shallower neighbor and the shared side.
        // Parent states have already been finalized by `ensure_node_state`.
        let face_parent = graph.parents(node).find_map(|(side, parent)| {
            graph.node_state(parent).megatree.tree.and_then(|state| {
                state
                    .child_sides
                    .contains(side)
                    .then_some((side, state.remaining_growth))
            })
        });
        let diagonal_parent = graph
            .parents(node)
            .flat_map(|(side, parent)| {
                graph
                    .node_state(parent)
                    .megatree
                    .outgoing()
                    .iter()
                    .filter(move |(outgoing_side, _)| *outgoing_side == side)
                    .map(|(_, signal)| *signal)
                    .filter(|signal| signal.is_complete())
            })
            .min_by_key(|signal| signal.key());
        let parent = face_parent
            .map(|(side, growth)| (Some(side), growth, SideSet::from_iter([side]), None))
            .or_else(|| {
                diagonal_parent.map(|signal| {
                    (
                        None,
                        signal.remaining_growth,
                        signal.target_sides,
                        Some((signal.source, signal.target_sides)),
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
                    let eligible_sides = diagonal_propagation_sides(tree_parent_sides)
                        .intersection(graph_child_sides(graph_parent_sides));
                    for side in Side::iter().filter(|side| eligible_sides.contains(*side)) {
                        if rng.random::<f32>() < branch_probability(enviro) {
                            child_sides.insert(side);
                        }
                    }
                } else {
                    let trunk_side = parent_side.unwrap_or_else(|| {
                        Side::iter().find(|s| incoming_sides.contains(*s)).unwrap()
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
        let has_leaves = state.tree.is_some_and(|tree| {
            tree.child_sides.is_empty()
                && !state
                    .diagonal
                    .as_ref()
                    .is_some_and(|diagonal| diagonal.has_children)
        });
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
    use crate::dodeca::Vertex;
    use crate::worldgen::WorldgenConfig;

    fn climate(rainfall: f32, temperature: f32) -> EnviroFactors {
        EnviroFactors {
            max_elevation: 0.0,
            rainfall,
            temperature,
            blockiness: 0.0,
        }
    }

    #[test]
    fn nodes_without_diagonal_signals_keep_compact_state() {
        // Diagonal vectors are boxed only for nodes that actually carry diagonal state.
        assert!(std::mem::size_of::<MegatreeState>() <= 24);
        assert!(
            MegatreeState::new(
                &Graph::new(1),
                NodeId::ROOT,
                NodeStateKind::Sky,
                climate(0.0, 0.0),
            )
            .diagonal
            .is_none()
        );
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
        MegatreeNode::new(
            &graph,
            child,
            kind,
            climate(0.0, 10.0),
            SideSet::from_iter([Side::B]),
            SideSet::from_iter([Side::B]),
        )
        .0
        .unwrap()
    }

    #[test]
    fn sky_branches_avoid_crossing_the_parent_face_plane() {
        // With probability 1, every face exit stays on the source side of the parent plane.
        let tree = inherited_tree(NodeStateKind::DeepSky);
        let child_sides = tree.child_sides;
        assert_eq!(child_sides.len(), 6);
        assert!(!child_sides.contains(Side::B));
        assert!(
            Side::iter()
                .filter(|side| { *side != Side::B && !side.adjacent_to(Side::B) })
                .all(|side| { child_sides.contains(side) })
        );
    }

    #[test]
    fn sky_branches_include_faces_adjacent_to_the_opposite_parent_face() {
        // These five face exits surround the face opposite the incoming parent.
        let parent_side = Side::B;
        let opposite_face = parent_side.opposite();
        let tree = inherited_tree(NodeStateKind::DeepSky);
        let adjacent_exits = Side::iter()
            .filter(|side| side.adjacent_to(opposite_face))
            .collect::<Vec<_>>();

        assert_eq!(adjacent_exits.len(), 5);
        assert!(
            adjacent_exits
                .iter()
                .all(|side| tree.child_sides.contains(*side))
        );
    }

    #[test]
    fn deep_land_always_continues_as_trunk() {
        // DeepLand always continues the trunk through the face opposite its incoming parent.
        let tree = inherited_tree(NodeStateKind::DeepLand);
        assert_eq!(tree.child_sides, SideSet::from_iter([Side::B.opposite()]));
    }

    #[test]
    fn edge_destinations_converge_to_one_tree_and_charge_growth_once() {
        // Both face-crossing orders deliver one canonical signal to the edge destination.
        let (graph, target) = graph_with_edge_carriers([Side::A, Side::B]);
        let state = MegatreeState::new(&graph, target, NodeStateKind::Sky, climate(100.0, 10.0));
        let tree = state.tree.unwrap();
        let mut rng =
            Pcg64Mcg::seed_from_u64(super::super::hash(graph.hash_of(target) as u64, SEED));
        let expected_growth =
            6.5 - 1.0 + rng.sample(Normal::new(0.0, GROWTH_COST_STANDARD_DEVIATION).unwrap());
        assert_eq!(tree.remaining_growth, expected_growth);
        assert_eq!(tree.parent_side, None);
        let source = transform_sides(SideSet::from_iter([Side::A, Side::B])) * MPoint::origin();
        assert!(
            state
                .diagonal
                .as_ref()
                .unwrap()
                .segments
                .iter()
                .any(|segment| {
                    segment.distance_to(&source) < 1e-5
                        && segment.distance_to(&MPoint::origin()) < 1e-5
                })
        );
        assert!(tree.remaining_growth > 0.0);
        assert_eq!(
            graph.parents(target).count(),
            2,
            "both carrier states exist before the target is finalized"
        );
    }

    #[test]
    fn edge_signal_is_independent_of_carrier_generation_order() {
        // Constructing the same edge carriers in reverse order preserves the final tree state.
        let (graph_ab, target_ab) = graph_with_edge_carriers([Side::A, Side::B]);
        let (graph_ba, target_ba) = graph_with_edge_carriers([Side::B, Side::A]);
        assert_eq!(target_ab, target_ba);
        let state_ab = MegatreeState::new(
            &graph_ab,
            target_ab,
            NodeStateKind::Sky,
            climate(100.0, 10.0),
        );
        let state_ba = MegatreeState::new(
            &graph_ba,
            target_ba,
            NodeStateKind::Sky,
            climate(100.0, 10.0),
        );
        let tree_ab = state_ab.tree.unwrap();
        let tree_ba = state_ba.tree.unwrap();
        assert_eq!(tree_ab.parent_side, tree_ba.parent_side);
        assert_eq!(
            state_ab.diagonal.as_ref().unwrap().segments.len(),
            state_ba.diagonal.as_ref().unwrap().segments.len()
        );
        assert_eq!(tree_ab.child_sides, tree_ba.child_sides);
        assert_eq!(tree_ab.remaining_growth, tree_ba.remaining_growth);
    }

    #[test]
    fn convergent_sources_choose_one_parent_by_stable_signal_key() {
        // Distinct branches reaching the same destination resolve to one stable parent.
        let (mut graph, target) = graph_with_edge_carriers([Side::A, Side::B]);
        let parent = graph.neighbor(target, Side::B).unwrap();
        graph[parent]
            .state
            .as_mut()
            .unwrap()
            .megatree
            .diagonal
            .as_mut()
            .unwrap()
            .outgoing
            .push((
                Side::B,
                CarrierSignal {
                    source: parent,
                    target_sides: SideSet::from_iter([Side::A, Side::B]),
                    crossed_sides: SideSet::from_iter([Side::A, Side::B]),
                    remaining_growth: 3.0,
                },
            ));

        let state = MegatreeState::new(&graph, target, NodeStateKind::Land, climate(100.0, 10.0));
        let tree = state.tree.unwrap();
        assert_eq!(tree.remaining_growth, INITIAL_REMAINING_GROWTH);
        assert_eq!(state.diagonal.as_ref().unwrap().segments.len(), 1);
    }

    #[test]
    fn edge_carrier_renders_segment_without_becoming_tree_node() {
        // A carrier forwards the signal and rasterizes its segment without becoming a tree node.
        let (graph, _) = graph_with_edge_carriers([Side::A, Side::B]);
        let carrier = graph.neighbor(NodeId::ROOT, Side::A).unwrap();
        let state = &graph.node_state(carrier).megatree;
        assert!(state.tree.is_none());
        assert_eq!(state.diagonal.as_ref().unwrap().segments.len(), 1);
        assert!(state.outgoing().iter().any(|(side, _)| *side == Side::B));
        let chunk = MegatreeChunk::new(state, 12, Vertex::A).unwrap();
        let mut voxels = VoxelData::Solid(Material::Void);
        chunk.generate(&mut voxels);
        let voxel_count = (12usize + 2).pow(3);
        assert!((0..voxel_count).any(|index| voxels.get(index) == Material::Wood));
        assert!((0..voxel_count).all(|index| voxels.get(index) != Material::Leaves));
    }

    #[test]
    fn negative_growth_at_diagonal_destination_stops_propagation() {
        // Growth is charged once on arrival; a negative result leaves a terminal crown.
        let (mut graph, target) = graph_with_edge_carriers([Side::A, Side::B]);
        for parent_side in [Side::A, Side::B] {
            let parent = graph.neighbor(target, parent_side).unwrap();
            let outgoing = &mut graph[parent]
                .state
                .as_mut()
                .unwrap()
                .megatree
                .diagonal
                .as_mut()
                .unwrap()
                .outgoing;
            for (_, signal) in outgoing.iter_mut().filter(|(side, _)| *side == parent_side) {
                signal.remaining_growth = 0.0;
            }
        }

        let state =
            MegatreeState::new(&graph, target, NodeStateKind::DeepSky, climate(100.0, 10.0));
        let tree = state.tree.unwrap();
        assert!(tree.remaining_growth < 0.0);
        assert!(tree.child_sides.is_empty());
        assert!(state.outgoing().is_empty());
        assert!(
            MegatreeChunk::new(&state, 12, Vertex::A)
                .unwrap()
                .has_leaves
        );
    }

    #[test]
    fn edge_branch_geometry_connects_source_and_destination_centers() {
        // The carrier's geodesic segment reaches the centers on either side of the edge.
        let target_sides = SideSet::from_iter([Side::A, Side::B]);
        let signal = CarrierSignal {
            source: NodeId::ROOT,
            target_sides,
            crossed_sides: SideSet::from_iter([Side::A]),
            remaining_growth: INITIAL_REMAINING_GROWTH,
        };
        let source = Side::A.reflection() * MPoint::origin();
        let target = Side::B.reflection() * MPoint::origin();
        let line = signal.segment();
        assert!(line.distance_to(&source) < 1e-5);
        assert!(line.distance_to(&target) < 1e-5);
    }

    #[test]
    fn vertex_routes_converge_and_are_independent_of_generation_order() {
        // Six crossing orders produce one vertex destination and one growth charge.
        let (graph_a, target_a) = graph_with_vertex_carriers(false);
        let (graph_b, target_b) = graph_with_vertex_carriers(true);
        assert_eq!(target_a, target_b);
        assert_eq!(graph_a.parents(target_a).count(), 3);
        assert_eq!(graph_b.parents(target_b).count(), 3);

        let tree_a = graph_a.node_state(target_a).megatree.tree.unwrap();
        let tree_b = graph_b.node_state(target_b).megatree.tree.unwrap();
        assert_eq!(
            graph_a
                .node_state(target_a)
                .megatree
                .diagonal
                .as_ref()
                .unwrap()
                .segments
                .len(),
            graph_b
                .node_state(target_b)
                .megatree
                .diagonal
                .as_ref()
                .unwrap()
                .segments
                .len()
        );
        assert_eq!(tree_a.child_sides, tree_b.child_sides);
        assert_eq!(tree_a.remaining_growth, tree_b.remaining_growth);

        let target_kind = graph_a.node_state(target_a).kind;
        let expected_growth = if matches!(target_kind, NodeStateKind::Sky | NodeStateKind::DeepSky)
        {
            let mut rng =
                Pcg64Mcg::seed_from_u64(super::super::hash(graph_a.hash_of(target_a) as u64, SEED));
            INITIAL_REMAINING_GROWTH - 1.0
                + rng.sample(Normal::new(0.0, GROWTH_COST_STANDARD_DEVIATION).unwrap())
        } else {
            INITIAL_REMAINING_GROWTH
        };
        assert_eq!(tree_a.remaining_growth, expected_growth);
    }

    #[test]
    fn vertex_branch_geometry_connects_source_and_destination_centers() {
        // A vertex signal's geodesic spans the already-crossed and remaining faces.
        let [a, b, c] = Vertex::A.canonical_sides();
        let signal = CarrierSignal {
            source: NodeId::ROOT,
            target_sides: SideSet::from_iter([a, b, c]),
            crossed_sides: SideSet::from_iter([a, b]),
            remaining_growth: INITIAL_REMAINING_GROWTH,
        };
        let source = transform_sides(SideSet::from_iter([a, b])) * MPoint::origin();
        let target = c.reflection() * MPoint::origin();
        let line = signal.segment();
        assert!(line.distance_to(&source) < 1e-5);
        assert!(line.distance_to(&target) < 1e-5);
    }

    #[test]
    fn land_seed_uses_side_a_as_ground_axis() {
        // A newly seeded Land tree begins along the ground-plane axis and its opposite face.
        let graph = Graph::new(1);
        let tree = MegatreeNode::new(
            &graph,
            NodeId::ROOT,
            NodeStateKind::Land,
            climate(100.0, 10.0),
            SideSet::EMPTY,
            SideSet::EMPTY,
        )
        .0
        .unwrap();
        assert_eq!(
            tree.child_sides,
            SideSet::from_iter([Side::A, Side::A.opposite()])
        );
        assert_eq!(tree.parent_side, None);
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
