#include "tp_caffe2_utils/Print.h"
#include "tp_caffe2_utils/BlobUtils.h"

#include "tp_utils/DebugUtils.h"

namespace tp_caffe2_utils
{

//##################################################################################################
void printBlob(caffe2::Workspace& workspace, const std::string& name)
{
  std::vector<float> data;
  readBlob(workspace, name, data);
  tpWarning() << std::fixed << data;
}

//##################################################################################################
void printArg(const caffe2::Argument& arg)
{
  std::string dbg = arg.DebugString();
  std::replace(dbg.begin(), dbg.end(), '\n', ' ');
  tpWarning() << "  Arg: " << dbg;
}

//##################################################################################################
void printOp(const caffe2::OperatorDef& op)
{
  tpWarning() << "Op: " << op.type();

  for(int i=0; i<op.input_size(); i++)
    tpWarning() << "  Input: " << op.input(i);

  for(int i=0; i<op.output_size(); i++)
    tpWarning() << "  Output: " << op.output(i);

  for(int i=0; i<op.arg_size(); i++)
    printArg(op.arg(i));
}

//##################################################################################################
void printOps(const caffe2::NetDef& net)
{
  for(int i=0; i<net.op_size(); i++)
    printOp(net.op(i));
}

}
