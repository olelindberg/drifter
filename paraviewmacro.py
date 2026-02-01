from paraview.simple import *

# Get the active source (the file you just opened)
source = GetActiveSource()

if source:
    # Add a Transform filter
    transform = Transform(Input=source)
    
    # Scale by 100 in the Z direction
    transform.Transform.Scale = [1.0, 1.0, 100.0]
    
    # Get the active view
    view = GetActiveViewOrCreate('RenderView')
    
    # Show the result and hide the original
    Hide(source, view)
    display = Show(transform, view)
    
    # Set representation to Surface With Edges
    display.SetRepresentationType('Surface With Edges')
    
    # Update the view
    Render()
    ResetCamera()