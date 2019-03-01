#include "tp_caffe2_utils/Globals.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void initCaffe2(int argc, char *argv[])
{
  caffe2::GlobalInit(&argc, &argv);
}

}
