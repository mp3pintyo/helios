// The displacement of a graph-displaced material in the extended resolve pipeline of its graph
// (replaces displacement_graphs_off.wgsl there): the pipeline only resolves materials of that
// graph, whose generated code defines `graph_displacement` (a height or a vector:
// GRAPH_DISPLACEMENT_VECTOR) next to `graph_eval`.

#include "displacement_graph.wgsl"

const DISPLACEMENT_VECTORS: bool = GRAPH_DISPLACEMENT_VECTOR;
const DISPLACEMENT_ANIMATED: bool = GRAPH_DISPLACEMENT_ANIMATED;

fn displacement_graph_value(graph: u32, material: u32, params: u32, max_scale: f32, p: DisplacementInput) -> DisplacementValue {
    let value = graph_displacement(materials[material], displacement_surface(material, max_scale, p), params);
    if GRAPH_DISPLACEMENT_VECTOR {
        return displacement_from_vector(vec3<f32>(value));
    }
    return displacement_from_height(vec3<f32>(value).x);
}
