
The code in this directory is mainly based on [XGBoost](https://github.com/dmlc/xgboost). Since the GBST package only uses the compiled library file and limited objective/metrics, the unrelated parts are trimmed.

To build the library from source, we start from making a working directory:

```
cd gbst_src
mkdir build
```

Then, for Windows:
```
cmake .. -G "Visual Studio 14 2015 Win64"
# for VS15: cmake .. -G"Visual Studio 15 2017" -A x64
# for VS16: cmake .. -G"Visual Studio 16 2019" -A x64
cmake --build . --config Release
```
If the build process successfully ends, you should find the library file **gbst.dll** in the ./gbst_src/lib folder.

For Linux:

```
cmake .. 
make -j$(nproc)
```

If the build process successfully ends, you should find the library file **libgbst.so** in the ./gbst_src/lib folder.

After that, please copy the compiled library into ../gbst_package/GBST/ , in place of the pre-compiled library file(s), then install the python package by previous instructions.