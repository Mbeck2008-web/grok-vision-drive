# ctest helper: run gvd_viz offline on a scene file and require a PNG.
file(REMOVE ${OUT})
execute_process(COMMAND ${EXE} --offline ${SCENE} --png ${OUT} --size 320x180 RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "gvd_viz exited ${rc}")
endif()
if(NOT EXISTS ${OUT})
  message(FATAL_ERROR "no PNG written")
endif()
