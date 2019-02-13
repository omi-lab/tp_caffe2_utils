TARGET = tp_caffe2_utils
TEMPLATE = lib

DEFINES += TP_CAFFE2_UTILS_LIBRARY

HEADERS += inc/tp_caffe2_utils/Globals.h

SOURCES += src/ArgHelpers.cpp
HEADERS += inc/tp_caffe2_utils/ArgHelpers.h

SOURCES += src/BlobHelpers.cpp
HEADERS += inc/tp_caffe2_utils/BlobHelpers.h

SOURCES += src/Ops.cpp
HEADERS += inc/tp_caffe2_utils/Ops.h

SOURCES += src/FillOps.cpp
HEADERS += inc/tp_caffe2_utils/FillOps.h

SOURCES += src/Print.cpp
HEADERS += inc/tp_caffe2_utils/Print.h

SOURCES += src/OpHelpers.cpp
HEADERS += inc/tp_caffe2_utils/OpHelpers.h

SOURCES += src/NetHelpers.cpp
HEADERS += inc/tp_caffe2_utils/NetHelpers.h
