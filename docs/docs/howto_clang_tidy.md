# How to use `clang-tidy`

`clang-tidy` is [a command-line tool](https://clang.llvm.org/extra/clang-tidy/) that automatically enforces certain aspects of code style and provides warnings for common programming mistakes. It automatically runs on every pull request in the Quokka GitHub repository.

## Using clang-tidy with VSCode

The easiest way to use `clang-tidy` on your own computer is to install the [clangd extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) (VSCode).

(VSCode itself can be downloaded [here](https://code.visualstudio.com/).)

## Command-line alternative

You can also run `clang-tidy` from the command line (see the [documentation](https://clang.llvm.org/extra/clang-tidy/#using-clang-tidy)).

To see the `clang-tidy` warnings that are relevant only to the code changes you've made, you can use the *clang-tidy-diff.py* [Python script](https://clang.llvm.org/extra/doxygen/clang-tidy-diff_8py_source.html).
