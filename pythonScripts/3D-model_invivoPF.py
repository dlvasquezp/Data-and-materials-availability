# -*- coding: utf-8 -*-
#!/usr/bin/env python

import vtk
from numpy import linspace as linspace
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(fileName1,fileName2):
    
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data.
    reader1 = vtk.vtkStructuredPointsReader()
    reader1.SetFileName(fileName1)
    
    reader2 = vtk.vtkStructuredPointsReader()
    reader2.SetFileName(fileName2)
    
    # Create transfer mapping scalar value to opacity.
    opacityTransferFunction1 = vtk.vtkPiecewiseFunction()
    opacityTransferFunction1.AddPoint(      0,  0.00)
    opacityTransferFunction1.AddPoint(     80,  0.00)
    opacityTransferFunction1.AddPoint(    128,  0.20)
    opacityTransferFunction1.AddPoint(    150,  0.00)
    opacityTransferFunction1.AddPoint(    255,  0.90)

    #Map color
    norm = mpl.colors.Normalize(vmin=120,vmax=255)
    cmap = plt.cm.Blues
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create transfer mapping scalar value to color.
    colorTransferFunction1 = vtk.vtkColorTransferFunction()
    for pixelVal in linspace(120,255):
        rgba = mapper.to_rgba(pixelVal)
        colorTransferFunction1.AddRGBPoint( pixelVal , rgba[0], rgba[1], rgba[2])
        
    #Map color
    norm = mpl.colors.Normalize(vmin=120,vmax=255)
    cmap = plt.cm.Greens
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create transfer mapping scalar value to color.
    colorTransferFunction2 = vtk.vtkColorTransferFunction()
    for pixelVal in linspace(120,255):
        rgba = mapper.to_rgba(pixelVal)
        colorTransferFunction2.AddRGBPoint( pixelVal , rgba[0], rgba[1], rgba[2])
    
    # Volume gradient     
    volumeGradientOpacity1 = vtk.vtkPiecewiseFunction()
    volumeGradientOpacity1.AddPoint(      0,  0.1)
    volumeGradientOpacity1.AddPoint(     20,  0.5)

    # The property describes how the data will look.
    volumeProperty1 = vtk.vtkVolumeProperty()
    volumeProperty1.SetColor(colorTransferFunction1)
    volumeProperty1.SetScalarOpacity(opacityTransferFunction1)
    volumeProperty1.SetInterpolationTypeToLinear()
    volumeProperty1.SetGradientOpacity(volumeGradientOpacity1)

    volumeProperty2 = vtk.vtkVolumeProperty()
    volumeProperty2.SetColor(colorTransferFunction2)
    volumeProperty2.SetScalarOpacity(opacityTransferFunction1)
    volumeProperty2.SetInterpolationTypeToLinear()
    volumeProperty2.SetGradientOpacity(volumeGradientOpacity1)
    
    # The mapper / ray cast function know how to render the data.
    volumeMapper1 = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper1.SetInputConnection(reader1.GetOutputPort())
    
    volumeMapper2 = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper2.SetInputConnection(reader2.GetOutputPort())
    
    # Set volumes
    volume1 = vtk.vtkVolume()
    volume1.SetMapper(volumeMapper1)
    volume1.SetProperty(volumeProperty1)
    
    volume2 = vtk.vtkVolume()
    volume2.SetMapper(volumeMapper2)
    volume2.SetProperty(volumeProperty2)
    
    # An outline provides context around the data.
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputConnection(reader1.GetOutputPort())

    mapOutline = vtk.vtkPolyDataMapper()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())

    outline = vtk.vtkActor()
    outline.SetMapper(mapOutline)
    outline.GetProperty().SetColor(colors.GetColor3d("Black"))

    # render volumes
    ren.AddVolume(volume1)
    ren.AddVolume(volume2)

    ren.SetBackground(colors.GetColor3d("White"))
    ren.GetActiveCamera().Roll(180)
    ren.GetActiveCamera().Azimuth(0)
    ren.GetActiveCamera().Elevation(230)
    ren.ResetCameraClippingRange()
    ren.ResetCamera()
    
    renWin.SetSize(800, 600)
    renWin.Render()
    
    iren.Start()

if __name__ == '__main__':
    
    fileName1 = '../data/VTKmodel/InvivoPF_Bundle1.vtk'
    fileName2 = '../data/VTKmodel/InvivoPF_Bundle2.vtk'
    
    main(fileName1,fileName2)