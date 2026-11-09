// The scene instance an instance index of the visibility buffer belongs to: itself, or for an
// assembly part (in the slots after the scene's instances, assembly.wgsl) its assembly's
// instance. Requires a `part_owners` buffer: [0] the scene's instance count, [1 + slot] the
// owner of part slot `slot` (assembly.rs; a scene without assemblies binds [0xffffffff]).

fn owner_instance(instance: u32) -> u32 {
    let static_count = part_owners[0];
    if instance == INVALID_ID || instance < static_count {
        return instance;
    }
    let slot = 1u + (instance - static_count);
    if slot >= arrayLength(&part_owners) {
        return INVALID_ID;
    }
    return part_owners[slot];
}
