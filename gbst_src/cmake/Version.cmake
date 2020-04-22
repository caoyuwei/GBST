function (write_version)
  message(STATUS "gbst VERSION: ${gbst_VERSION}")
  configure_file(
    ${gbst_SOURCE_DIR}/cmake/version_config.h.in
    ${gbst_SOURCE_DIR}/include/xgboost/version_config.h @ONLY)
endfunction (write_version)
