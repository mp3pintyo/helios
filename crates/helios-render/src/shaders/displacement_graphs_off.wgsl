// Passes without material graphs that drive displacement (displacement_height.wgsl). The
// tessellation passes of a scene with such graphs get a generated module in place of this file
// (tessellation/graphs.rs: every displacing graph's code and a switch on the material's graph),
// the extended resolve of a displacing graph displacement_graph_resolve.wgsl. Elsewhere no
// material is graph-displaced: its surface stays at the centre height.

// The resolve's graph moves its displaced surfaces with the time (their motion vectors then need
// the previous frame's surface).
const DISPLACEMENT_ANIMATED: bool = false;

fn displacement_graph_value(graph: u32, material: u32, params: u32, max_scale: f32, p: DisplacementInput) -> DisplacementValue {
    return displacement_from_height(tess.materials[material].params.y);
}
