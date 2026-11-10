// The height of a graph-displaced material in the extended resolve pipeline of its graph
// (replaces displacement_graphs_off.wgsl there): the pipeline only resolves materials of that
// graph, whose generated code defines `graph_displacement` next to `graph_eval`.

#include "displacement_graph.wgsl"

fn displacement_graph_height(graph: u32, material: u32, params: u32, max_scale: f32, p: DisplacementInput) -> f32 {
    return graph_displacement(materials[material], displacement_surface(material, max_scale, p), params);
}
