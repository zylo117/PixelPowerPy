# sfr_circle
#   Calculates the SFR score for every circle patch in the SFR image
#
# INPUT:
#   IDraw:  Bayer-channel of an image
#   varargin : inputStruct of your program with all required settings
#
# OUTPUT:
#       output : Output struct containing everything you want
#           OC_X/Y
#           Ny8/Ny4/Ny2/Acc
#               Cen/30F/60F/75F/Edge
#                   Min/Max/Avg/StdDev
#                       Tan/Sag
#               Inner/Mid/Edge_Delta
#
#           If debugFlag is enabled, every ROI has the following output
#               Ny8/Ny4/Ny2/Acc/X/Y/Radius

def sfr_circle(IDraw, *args):

    inputParameters = args[0]
    inputParamaterSectionName = args[1]

    dotInfo = inputParameters[inputParamaterSectionName]
    fieldPoints = dotInfo["fieldPoints"]

    bayerFormat = inputParameters["sensor"]["bayerFormat"]
    FOV = inputParameters[inputParamaterSectionName]["fov"]
    pedestal = inputParameters["sensor"]["pedestal"]
    bitDepth = inputParameters["sensor"]["bitDepth"]
    dotInfo.seed = dotInfo["chartIsHalfGrid"]
    chartType = dotInfo["chartType"]
    plotFlag = dotInfo["plotFlag"]
