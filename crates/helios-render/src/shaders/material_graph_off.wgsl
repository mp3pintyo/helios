// Extended resolve pipelines of materials without a graph (Substrate slabs only): the
// material's own shading model runs and `graph_eval` is never called. The pipelines of graph
// materials get the graph's generated code (helios-material) in place of this file.

const HAS_GRAPH: bool = false;
const GRAPH_OUTPUTS: u32 = 0u;

fn graph_eval(m: Material, input: SurfaceInput, params: u32, vertex_color: vec4<f32>, defaults: GraphOutput) -> GraphOutput {
    return defaults;
}
