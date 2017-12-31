"""
 SFR Calculates the SFR score for every patch in the SFR image

 INPUT:
   IDraw:  Bayer-channel of an image
   bayerFormat: Bayer format of the image
   FOV: FOV of the camera lens
   pedestal: pedestal of the image
   bitDepth: bitDepth of the image
   program: name of the program
   relay: Flag set to 1 for relay lenses with offset center ROI
   accutanceCutoff: Frequency to integrate SFR per ROI
   debug: Flat set to 1 for image markup

 OUTPUT:
   Max, Min, Mean, Standard Deviation, and Uniformity (STD / Man) of the
   following trends for the [Center, 30F, 60F, 75F, EDGE] ROIs:
       SFR Nyquist/8
       SFR Nyquist/4
       SFR Accutance
       AS Edge Nyquist/8
       AS Edge Nyquist/4
       AS Edge Mean

IDraw

 COMMENTS:

   Extra ROIs can be removed according to ERS by commenting them out 
"""
