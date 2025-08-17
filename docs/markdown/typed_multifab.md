# TypedMultifab: Type-Safe AMReX MultiFab Wrapper

## Overview

TypedMultifab is a type-safe wrapper around AMReX MultiFabs that provides compile-time checking and strongly-typed access to array components. It eliminates the need for manual index management and enables zero-copy construction from existing MultiFabs.

## Key Features

- **Type Safety**: Access components using strongly-typed variable names instead of integer indices
- **Zero-Copy Construction**: Create subsets or combine TypedMultifabs without deep copying data
- **Multi-Component Variables**: Support for variables with multiple components (e.g., scalar fields)
- **AMReX Compatibility**: Extract underlying MultiFabs for seamless integration with existing AMReX code
- **Compile-Time Validation**: All component access is checked at compile time

## Basic Usage

### Defining Variables

Single-component variables are defined using the `VARIABLE` macro:

```cpp
namespace Conserved
{
VARIABLE(hydro, density);
VARIABLE(hydro, momentum_x);
VARIABLE(hydro, momentum_y);
VARIABLE(hydro, momentum_z);
VARIABLE(hydro, energy);
} // namespace Conserved
```

Multi-component variables use the `MULTI_VARIABLE` macro:

```cpp
namespace Conserved
{
// Define a scalar variable with 10 components
constexpr int NumScalars = 10;
MULTI_VARIABLE(hydro, scalar, NumScalars);
} // namespace Conserved
```

### Creating Strong Type Aliases

For multi-component variables, you can create strongly-typed aliases for individual components:

```cpp
namespace ProblemSpecific
{
// Create meaningful names for specific scalar components
COMPONENT_ALIAS(Conserved::scalar, 0, temperature);
COMPONENT_ALIAS(Conserved::scalar, 1, metallicity);
COMPONENT_ALIAS(Conserved::scalar, 2, electron_fraction);
} // namespace ProblemSpecific
```

### Defining Type Lists

TypeLists specify which variables a TypedMultifab contains:

```cpp
// Basic type list
using ConservedHydroOnly = quokka::TypeList<
    Conserved::density,
    Conserved::momentum_x,
    Conserved::momentum_y,
    Conserved::momentum_z,
    Conserved::energy
>;

// Expanding multi-component variables
using ConservedScalarsExpanded = quokka::ExpandMultiVariable_t<Conserved::scalar>;

// Combining type lists
using ConservedTypeList = quokka::TypeListCat_t<ConservedHydroOnly, ConservedScalarsExpanded>;
```

## Creating TypedMultifabs

### Standard Construction

Create a new TypedMultifab with all components in a single underlying MultiFab:

```cpp
const amrex::BoxArray &ba = ...;
const amrex::DistributionMapping &dm = ...;
const int nghost = ...;

quokka::TypedMultifab<ConservedTypeList> conserved_mf(ba, dm, nghost);
```

### Zero-Copy Subset Construction

Create a TypedMultifab that references components from an existing TypedMultifab:

```cpp
// Original TypedMultifab with all conserved variables
quokka::TypedMultifab<ConservedTypeList> conserved_mf(ba, dm, nghost);

// Create subset with only hydro variables (no copy)
quokka::TypedMultifab<ConservedHydroOnly> hydro_mf(ba, dm, nghost, conserved_mf);

// Create subset with only first 3 scalar components (no copy)
using ScalarsSubset = quokka::TypeList<Conserved::scalar<0>, Conserved::scalar<1>, Conserved::scalar<2>>;
quokka::TypedMultifab<ScalarsSubset> scalar_mf(ba, dm, nghost, conserved_mf);
```

### Combining Multiple TypedMultifabs

Combine components from multiple TypedMultifabs without copying:

```cpp
// Create combined TypedMultifab from multiple sources
using CombinedList = quokka::TypeListCat_t<ConservedHydroOnly, ScalarsSubset>;
auto combined_mf = quokka::makeTypedMultifab<CombinedList>(ba, dm, nghost, hydro_mf, scalar_mf);
```

## Accessing Components

### Array Access in Kernels

Access components using type-safe array access:

```cpp
for (amrex::MFIter mfi(typed_state.getMultiFab<Conserved::density>()); mfi.isValid(); ++mfi) {
    const amrex::Box &bx = mfi.validbox();
    
    // Get typed arrays for single-component variables
    auto density_arr = typed_state.array<Conserved::density>(mfi);
    auto momx_arr = typed_state.array<Conserved::momentum_x>(mfi);
    auto energy_arr = typed_state.array<Conserved::energy>(mfi);
    
    // Access multi-component variables using strong types
    auto temp_arr = typed_state.array<ProblemSpecific::temperature>(mfi);
    auto metal_arr = typed_state.array<ProblemSpecific::metallicity>(mfi);
    
    // Access multi-component variables by index
    auto scalar5_arr = typed_state.array<Conserved::scalar<5>>(mfi);
    
    // Use in parallel kernel
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        density_arr(i, j, k) = 1.0;
        temp_arr(i, j, k) = 300.0;
        // No manual indexing needed!
    });
}
```

### Component Information

Query component names and properties:

```cpp
// Get all component names
std::vector<std::string> names = typed_mf.component_names();
// Returns: ["hydro.density", "hydro.momentum_x", ..., "hydro.scalar[0]", "hydro.scalar[1]", ...]

// Get specific component name
std::string name = quokka::TypedMultifab<ConservedTypeList>::component_name<Conserved::density>();
// Returns: "hydro.density"

// Get name of multi-component variable element
std::string scalar_name = ProblemSpecific::temperature::name();
// Returns: "hydro.scalar[0]"

// Check if component exists
bool has_density = typed_mf.hasComponent<Conserved::density>();

// Get number of components
constexpr std::size_t ncomp = ConservedTypeList::num_components();
```

### AMReX Interoperability

Extract underlying MultiFabs for use with AMReX functions:

```cpp
// Get MultiFab containing a specific component
amrex::MultiFab &density_mf = typed_mf.getMultiFab<Conserved::density>();

// Get component index within its MultiFab
int density_idx = typed_mf.getComponentIndex<Conserved::density>();
```

### Generic Component Processing

For operations that need to process all components identically (e.g., reconstruction, interpolation), use the contiguous component iterator:

```cpp
// Get groups of contiguous components
auto component_groups = typed_mf.getContiguousComponentGroups();

// Process each group of contiguous components
for (const auto &group : component_groups) {
    // Create an alias MultiFab for this group (zero-copy)
    amrex::MultiFab alias_mf = TypedMultifab<ConservedTypeList>::makeAliasMultiFab(group);
    
    // Information about the group
    std::cout << "Processing " << group.num_comp << " contiguous components:\n";
    for (const auto &name : group.component_names) {
        std::cout << "  - " << name << "\n";
    }
    
    // Apply generic operation to all components in the group
    for (amrex::MFIter mfi(alias_mf); mfi.isValid(); ++mfi) {
        const amrex::Box &bx = mfi.validbox();
        auto const &arr = alias_mf.array(mfi);
        
        // Example: Apply reconstruction to all components
        amrex::ParallelFor(bx, group.num_comp,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
            // 'n' iterates from 0 to group.num_comp-1
            // Access component as arr(i,j,k,n)
            
            // Example: Simple linear reconstruction
            Real left = 0.5 * (arr(i-1,j,k,n) + arr(i,j,k,n));
            Real right = 0.5 * (arr(i,j,k,n) + arr(i+1,j,k,n));
            // Store results somewhere...
        });
    }
}
```

This approach is particularly useful for:
- **Reconstruction algorithms**: Apply the same reconstruction to all conserved variables
- **Limiters**: Apply slope limiters uniformly across components
- **Boundary conditions**: Apply the same BC logic to multiple components
- **Copy operations**: Copy all components between different MultiFabs
- **Interpolation**: Interpolate all components during AMR operations

The iterator automatically identifies which components are stored contiguously in memory, enabling efficient vectorized operations while maintaining the type-safe interface.

## Advanced Features

### Type List Operations

TypedMultifab provides several compile-time operations on type lists:

```cpp
// Check if type is in list
constexpr bool has_density = quokka::TypeListContains_v<Conserved::density, ConservedTypeList>;

// Get type at specific index
using SecondType = ConservedTypeList::type<1>;  // Conserved::momentum_x

// Get index of type
constexpr std::size_t density_idx = ConservedTypeList::GetIdx<Conserved::density>();

// Create sublists
using FirstThree = ConservedTypeList::sublist<0, 1, 2>;

// Iterate over types at compile time
ConservedTypeList::IterateTypes([](auto t) {
    using VarType = decltype(t);
    std::cout << VarType::name() << "\n";
});
```

### Multi-Component Variable Patterns

Working with multi-component variables:

```cpp
// Check if variable is multi-component
constexpr bool is_multi = quokka::IsMultiComponent_v<Conserved::scalar<0>>;

// Access all components of a multi-component variable
for (int i = 0; i < Conserved::scalar::num_components; ++i) {
    // Note: This requires runtime indexing, prefer compile-time access when possible
}

// Expand multi-component variable into TypeList
using ExpandedScalars = quokka::ExpandMultiVariable_t<Conserved::scalar>;
// Equivalent to: TypeList<scalar<0>, scalar<1>, ..., scalar<9>>
```

## Design Rationale

### Memory Efficiency

TypedMultifab uses a component mapping system that tracks:
- Pointer to the underlying MultiFab (non-owning)
- Component index within that MultiFab
- Ownership flag

This design enables:
- Zero-copy subset construction
- Flexible component distribution across multiple MultiFabs
- Memory-efficient views of existing data

### Type Safety Benefits

1. **Compile-Time Errors**: Accessing non-existent components results in compile-time errors
2. **Self-Documenting Code**: Variable names make code intent clear
3. **Refactoring Safety**: Changing variable definitions automatically propagates through the codebase
4. **No Manual Indexing**: Eliminates off-by-one errors and index mismatches

### Performance

TypedMultifab has zero runtime overhead compared to manual MultiFab access:
- All type resolution happens at compile time
- Array access compiles to direct memory access
- No virtual functions or runtime polymorphism

## Migration from Index-Based Access

TypedMultifab provides a seamless migration path from traditional index-based MultiFab access. The AMRSimulation class automatically provides typed views alongside the existing interface:

```cpp
// In your simulation class
template <> 
void QuokkaSimulation<MyProblem>::setInitialConditions() 
{
    // Continue using existing index-based code
    state_new_cc_[lev][mfi](i, j, k, HydroSystem<MyProblem>::density_index) = 1.0;
    
    // Create typed views (zero-copy operation)
    syncTypedMultiFabs<ConservedTypeList>(lev);
    
    // Access typed state vectors
    auto& typed_state = (*getTypedStateNew<ConservedTypeList>())[lev];
    auto density_arr = typed_state.array<Conserved::density>(mfi);
    // Use typed arrays...
}
```

### Migration Steps

1. **Define Your Type Lists**: Create type lists matching your existing component layout
2. **Create Typed Views**: Call `syncTypedMultiFabs<TypeList>(lev)` to create zero-copy typed views
3. **Gradual Conversion**: Convert kernels one at a time from index-based to type-based access
4. **Remove Old Code**: Once fully migrated, remove index-based access patterns

## Best Practices

1. **Define Variables in Headers**: Place variable definitions in header files accessible to all code that needs them

2. **Use Strong Types**: Create meaningful aliases for multi-component variables in problem-specific namespaces

3. **Prefer Compile-Time Access**: Use template parameters for component access rather than runtime indexing

4. **Leverage Zero-Copy**: Create views and subsets instead of copying data when possible

5. **Document Component Meanings**: Add comments explaining what each component represents, especially for multi-component variables

6. **Gradual Migration**: Use the migration infrastructure to transition incrementally from index-based to type-based access

## Example: Complete Workflow

```cpp
// 1. Define variables
namespace State
{
VARIABLE(hydro, density);
VARIABLE(hydro, momentum_x);
VARIABLE(hydro, momentum_y);
VARIABLE(hydro, momentum_z);
VARIABLE(hydro, energy);

constexpr int NumTracers = 5;
MULTI_VARIABLE(hydro, tracer, NumTracers);
}

namespace MyProblem
{
COMPONENT_ALIAS(State::tracer, 0, dye_concentration);
COMPONENT_ALIAS(State::tracer, 1, metal_fraction);
}

// 2. Create type lists
using StateVars = quokka::TypeListCat_t<
    quokka::TypeList<State::density, State::momentum_x, State::momentum_y, 
                     State::momentum_z, State::energy>,
    quokka::ExpandMultiVariable_t<State::tracer>
>;

// 3. Use in simulation
void simulate() {
    // Create typed multifab
    quokka::TypedMultifab<StateVars> state(ba, dm, nghost);
    
    // Initialize with type-safe access
    for (amrex::MFIter mfi(state.getMultiFab<State::density>()); mfi.isValid(); ++mfi) {
        auto rho = state.array<State::density>(mfi);
        auto dye = state.array<MyProblem::dye_concentration>(mfi);
        
        const amrex::Box &bx = mfi.validbox();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = 1.0;
            dye(i, j, k) = (i < 50) ? 1.0 : 0.0;
        });
    }
}
```

## See Also

- [AMReX MultiFab Documentation](https://amrex-codes.github.io/amrex/docs_html/Basics.html#multifab)
- [Quokka Variable System](variables.md)
- [TypedMultifab Example](../../src/problems/TypedMultifabExample/test_typed_multifab.cpp)