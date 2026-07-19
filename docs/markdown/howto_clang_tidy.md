# How to use `clang-tidy`

`clang-tidy` is [a command-line tool](https://clang.llvm.org/extra/clang-tidy/) that automatically enforces certain aspects of code style and provides warnings for common programming mistakes. It automatically runs on every pull request in the Quokka GitHub repository.

## Using clang-tidy with VSCode

The easiest way to use `clang-tidy` on your own computer is to install the [clangd extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) (VSCode).

(VSCode itself can be downloaded [here](https://code.visualstudio.com/).)

## Command-line tools

The clangd extension for VSCode might not check all warnings, so you can also run `clang-tidy` from the command line (see the [documentation](https://clang.llvm.org/extra/clang-tidy/#using-clang-tidy)). Here is a minimal example of how to use it:

```bash
clang-tidy src/particles/* -p ./build
```

which will run `clang-tidy` on all the files in the `src/particles` directory.

To see the `clang-tidy` warnings that are relevant only to the code changes you've made, you can use `scripts/tidy.sh`. Example usage:

```bash
./scripts/tidy.sh ./build
```

This will run `clang-tidy` on all the files that have been modified with respect to the previous commit.

```bash
./scripts/tidy.sh ./build previous
```

This will run `clang-tidy` on all the files that have been modified in the last commit.

```bash
./scripts/tidy.sh ./build origin
```

This will run `clang-tidy` on all the files that have been modified with respect to the remote branch.

```bash
./scripts/tidy.sh ./build dev
```

This will run `clang-tidy` on all the files that have been modified with respect to the local development branch.

To see the `clang-tidy` warnings that are relevant only to individual lines of code you've changed, you can use the *clang-tidy-diff.py* [Python script](https://clang.llvm.org/extra/doxygen/clang-tidy-diff_8py_source.html).
