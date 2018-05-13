# focusGains image xtalk compensation

# Xtalk（crosstalk）induced two types effects： delay and static noise.

# crosstalk delay effect: crosstalk can affect signal delays by changing the times at which signal transitions occur,
# because of capacitive corss-coupling. this delay change maybe cause the transition to occur later or earlier,
# possibly contributing to a setup or hold violaion for a path.

# crosstalk noise effect: due to crosstalk, a noise bump happens on cross-coupled nets,
# which on steady-state(0 or 1). If the bump is sufficiently large and wide,
# it can cause an incorrect logic value to be propagated to the next gate.

# INPUT:
#   ID          - Input raw image
#   bayerFormat - bayer format of the raw image
#   pedestal    - Data pedestal of the input image
#   bitDepth    - Bitdepth of the input image
#   roiSize     - The ROI that defines the focus pixel repeating unit size
#   ROIs        - The number of repeating focus pixel unit ROIs in the
#   Kernels     - The [Kernel_L Kernel_R] that defines focus pixel placement
#   offset      - The [offsetX offsetY] that defines the first pixel of the
#                   first ROI 
#   medianArea  - [medianAreaX_Gain medianAreaY_Gain; medianAreaX_xtlk medianAreaY_xtlk] 
#                   defines focus pixel median areas
#   outputNVM   - Size of the output NVM grid of L/R focus pixels and
#                   crosstalk
#   
# OUTPUT:
#   IDrawComp   - Compensated image based on 4 nearby Gr/Gb average

def focusGains_OCL_xtlk_Comp(IDraw, bayerFormat, pedestal, bitDepth, roiSize, ROIs, Kernels, offset, isFPCGain, isFPCXTlk):
    h, w = IDraw.shape

    # Preprocess to perform pedestal subtraction and nothing else
    IDrawComp = IDraw

    # Initial output image is a copy of the input image

    # Extract shielded pixel information
    roiX = roiSize[0]
    roiY = roiSize[1]
    ROIsX = ROIs[0]
    ROIsY = ROIs[1]
    offsetX = offset[0]
    offsetY = offset[1]

    # Extract the focus pixels in the defined kernel
    CordsY = Kernels[:, 0]
    CordsX = Kernels[:, 1]
    
    # Skip in 'unit cell' blocks starting at the offsets such that each
    # unit cell has the predetermined offsets as per design documentation
    currentRow = offsetY; currentCol = offsetX


    #...