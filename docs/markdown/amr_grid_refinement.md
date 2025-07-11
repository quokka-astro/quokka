# Grid Refinement in Berger-Collela AMR

This document explains how Adaptive Mesh Refinement (AMR) works in Quokka, which uses the Berger-Collela AMR algorithm implemented in AMReX. Understanding these concepts is crucial for setting up AMR simulations effectively and avoiding common pitfalls.

## Overview of Berger-Collela AMR

Quokka uses the Berger-Collela AMR algorithm [@berger1989local] as implemented in the AMReX library. This algorithm automatically creates and manages a hierarchy of nested grids with different spatial resolutions to efficiently resolve features of interest while maintaining computational efficiency.

### Key Concepts

The AMR algorithm works by:

1. **Tagging cells** for refinement based on user-defined criteria (implemented in the `ErrorEst` function)
2. **Clustering tagged cells** into rectangular boxes using the Berger-Rigoutsos algorithm [@berger1991algorithm]
3. **Creating new grids** at the next refinement level based on these clusters
4. **Managing the grid hierarchy** across multiple refinement levels

## AMR Parameters

Several key parameters control the grid generation process:

| Parameter | Description | Typical Values | Performance Impact |
|-----------|-------------|----------------|-------------------|
| `blocking_factor` | Minimum grid size; grids must be divisible by this value | 8-32 | Higher values improve GPU performance but may cause over-refinement |
| `grid_efficiency` | Minimum fraction of refined cells in each grid | 0.7-1.0 | Higher values create more efficient grids but may increase memory usage |
| `max_grid_size` | Maximum size of individual grids | 64-128 | Affects load balancing and memory locality |
| `ref_ratio` | Refinement ratio between levels | 2 or 4 | Factor by which resolution increases at each level |

## Common Unintuitive Behaviors

### Whole Domain Refinement

**Problem**: When setting `amr.max_level = 1`, you may observe that the entire computational domain gets refined to level 1, even when only a small region should be refined according to your refinement criteria.

**Cause**: This happens due to the interaction between several AMR parameters:

1. **Blocking Factor Constraint**: All grids must have dimensions that are multiples of `blocking_factor`. If your `blocking_factor` is large (e.g., 32 or 64), and you have scattered refinement tags, the algorithm may need to create grids that cover most of the domain to satisfy this constraint.

2. **Grid Efficiency**: The algorithm tries to maintain a minimum `grid_efficiency` (fraction of tagged cells in each grid). Low efficiency may cause the algorithm to include many untagged cells.

3. **Coarsening Effect**: The AMReX algorithm first coarsens the tag list by `blocking_factor/ref_ratio` before applying the Berger-Rigoutsos clustering algorithm. This can cause small, isolated regions to merge into larger blocks.

**Example**: In the SphericalCollapse test problem, when `blocking_factor = 64` and the domain is 128³ cells, even a small spherical region requiring refinement can result in the entire domain being refined because the algorithm cannot create efficient grids smaller than the blocking factor.

**Solutions**:
- Reduce `blocking_factor` to allow finer control (minimum value is typically 4, constrained by `n_ghost`)
- Increase `grid_efficiency` closer to 1.0 to avoid including many untagged cells
- Consider the trade-off between refinement precision and computational performance

### Performance vs. Precision Trade-offs

**GPU Performance**: On GPUs, `blocking_factor` should typically be ≥32 for good performance. This conflicts with the desire for precise refinement.

**CPU Performance**: Even on CPUs, `blocking_factor` should be ≥8 for efficient multigrid operations in the Poisson solver.

**Memory Efficiency**: Large blocking factors can lead to significant memory overhead when refinement is only needed in small regions.

### Grid Efficiency Interactions

The `grid_efficiency` parameter can cause unexpected behavior:

- **Low efficiency** (< 0.8): May include many untagged cells, leading to unnecessary computation
- **High efficiency** (> 0.95): May create many small, inefficient grids that hurt performance
- **Default value** (0.7): Usually provides a good balance but may still cause over-refinement in some cases

## Understanding the Refinement Process

### Error Estimation and Tagging

The refinement process begins with the `ErrorEst` function, which is called by AMReX to determine which cells should be tagged for refinement. This is a virtual function that gets called indirectly through the AMReX grid management routines:

- `AmrCore::InitFromScratch()` during initialization
- `AmrCore::regrid()` during runtime regridding
- `AmrMesh::MakeNewGrids()` as part of the grid generation process

### Grid Generation Algorithm

1. **Tag Collection**: Cells meeting refinement criteria are tagged
2. **Coarsening**: Tags are coarsened by `blocking_factor/ref_ratio`
3. **Clustering**: The Berger-Rigoutsos algorithm groups tagged cells into rectangular boxes
4. **Grid Creation**: New grids are created from these boxes, respecting `blocking_factor` and `max_grid_size` constraints
5. **Efficiency Check**: Grids not meeting `grid_efficiency` requirements may be merged or extended

### Averaging Down Effects

After creating refined grids, the solution is averaged down from fine to coarse levels. This can improve the solution on coarse grids, potentially changing the refinement criteria on subsequent regridding steps. This feedback effect can lead to expanding or contracting refinement regions over time.

## Best Practices

### For Test Problems

- Use `blocking_factor = 4` or `blocking_factor = 8` to demonstrate proper refinement behavior
- Set `grid_efficiency = 1.0` for maximum precision when performance is not critical
- Monitor the actual grid structure in output files to verify expected refinement patterns

### For Production Runs

- Balance `blocking_factor` based on your computational platform:
  - GPU: ≥32 for performance
  - CPU: ≥8 for multigrid efficiency
- Test different `grid_efficiency` values to find the optimal balance for your problem
- Consider the memory and computational overhead of over-refinement

### Debugging Refinement Issues

When debugging unexpected refinement behavior:

1. **Check grid structure**: Examine plotfile headers or use visualization tools to see actual grid layout
2. **Reduce blocking_factor**: Temporarily use smaller values to verify refinement criteria
3. **Monitor efficiency**: Check if grids meet your `grid_efficiency` requirements
4. **Visualize tags**: Output refinement tags to understand what the `ErrorEst` function is actually selecting

## Detection and Warnings

Consider implementing detection for whole-domain refinement scenarios, especially during initialization. When the entire domain is refined to level 1, it often indicates:

- `blocking_factor` is too large for the problem size
- Refinement criteria are too broad
- Grid efficiency settings are suboptimal

Such detection can help users identify and correct these parameter mismatches early in their simulations.

## Further Reading

For more detailed information about the Berger-Collela algorithm:

- Berger & Colella (1989): "Local adaptive mesh refinement for shock hydrodynamics" [@berger1989local]
- Berger & Rigoutsos (1991): "An Algorithm for Point Clustering and Grid Generation" [@berger1991algorithm]
- [AMReX-Astro Castro documentation on regridding](https://amrex-astro.github.io/Castro/docs/regridding.html)
- [AMReX source code documentation](https://github.com/AMReX-Codes/amrex)

## References

The behavior described in this document was identified and discussed in [GitHub issue #978](https://github.com/quokka-astro/quokka/issues/978).