# CLAUDE.md - Guide for Working with Quokka Codebase

## Build & Test Commands
- Build: `mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja && ninja -j6`
- GPU Support: Add `-DAMReX_GPU_BACKEND=CUDA` (NVIDIA) or `-DAMReX_GPU_BACKEND=HIP` (AMD)
- Run all tests: `ctest` or `ninja test`
- Run specific test: `ctest -R TestName`
- Exclude tests: `ctest -E "Pattern*"`
- List test targets: `cmake --build . --target help`

## Code Style Guidelines
- Use `.clang-format` from `src/` directory for formatting (LLVM-based style)
- 160 character line limit, 8-space indentation with tabs
- Classes use PascalCase (e.g., `QuokkaSimulation`)
- Member variables use camelCase with trailing underscore (e.g., `radiationCflNumber_`)
- Always use curly braces for single statement blocks
- Document APIs using Doxygen style comments
- PRs should be focused on a single change and target the `development` branch
- Static analysis with clang-tidy available for code quality checks