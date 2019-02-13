#include "tp_caffe2_utils/OpUtils.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void removeOpByOutput(caffe2::NetDef& net,const std::string& opOutputName)
{
  for(int i=0; i<net.op_size(); i++)
  {
    auto op = net.op(i);

    bool remove=false;
    for(int j=0; j<op.output_size(); j++)
    {
      if(op.output(j) == opOutputName)
      {
        remove = true;
        break;
      }
    }

    if(remove)
    {
      op.Clear();
      break;
    }
  }
}

}
