TARGET = tp_caffe2_utils
TEMPLATE = lib

DEFINES += TP_CAFFE2_UTILS_LIBRARY

HEADERS += inc/tp_caffe2_utils/Globals.h

SOURCES += src/ArgUtils.cpp
HEADERS += inc/tp_caffe2_utils/ArgUtils.h

SOURCES += src/BlobUtils.cpp
HEADERS += inc/tp_caffe2_utils/BlobUtils.h

SOURCES += src/Ops.cpp
HEADERS += inc/tp_caffe2_utils/Ops.h

SOURCES += src/FillOps.cpp
HEADERS += inc/tp_caffe2_utils/FillOps.h

SOURCES += src/Print.cpp
HEADERS += inc/tp_caffe2_utils/Print.h

SOURCES += src/OpUtils.cpp
HEADERS += inc/tp_caffe2_utils/OpUtils.h

SOURCES += src/NetUtils.cpp
HEADERS += inc/tp_caffe2_utils/NetUtils.h
