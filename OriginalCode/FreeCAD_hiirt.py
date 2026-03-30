import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.0\lib')
sys.path.append(r'C:\Program Files\FreeCAD 1.0\bin')
import FreeCAD as App
import Part
import math
import random
import os
 
# Creazione di un nuovo documento
doc = App.newDocument("Girante")
 
# Dati utensile
tool_base = 40
tool_radius = 1
tool_depth = 8.3
num_teeth = 48
external_diameter = 200
internal_diameter = 150
hole_diameter = 50
diametro_base = 250
h_base = 10
h_master = 50
crown_height = 15
 
gamma = 60
alfa = 3.25
delta = 0
teta = 0
ym = 0
xm = 0
txm = (360 / 6.28) * xm / external_diameter
 
# Creazione della base cilindrica
base = Part.makeCylinder(diametro_base / 2, h_base)
doc.addObject("Part::Feature", "Base").Shape = base
 
# Creazione della seconda sezione cilindrica
second_section = Part.makeCylinder(external_diameter / 2, h_master)
second_section.translate(App.Vector(0, 0, h_base))
doc.addObject("Part::Feature", "SecondSection").Shape = second_section
 
# Creazione della corona dentata
outer_cylinder = Part.makeCylinder(external_diameter / 2, crown_height)
inner_cylinder = Part.makeCylinder(internal_diameter / 2, crown_height)
crown = outer_cylinder.cut(inner_cylinder)
crown.translate(App.Vector(0, 0, h_master + h_base))
doc.addObject("Part::Feature", "Crown").Shape = crown
 
# Creazione del foro centrale
center_hole = Part.makeCylinder(hole_diameter / 2, 2 * h_master)
body_with_hole = base.fuse(second_section).fuse(crown).cut(center_hole)
doc.addObject("Part::Feature", "BodyWithHole").Shape = body_with_hole
 
doc.recompute()
 
# Posizionamento dell'utensile e taglio dei denti
teeth_angle = 360 / num_teeth
crown_with_teeth = body_with_hole
 
for i in range(num_teeth):
    gamma = 60 + 0*random.gauss(59.994, 0.003) #implementato
    xm = 2 + 0*random.gauss(0.004, 0.001) #implementato
    ym = 2 + 0*random.gauss(0.037, 0.001) #implementato
    alfa = 8 + 0*random.gauss(3.252, 0.001)#implementato
    teta = 8 + 0*random.gauss(-0.012, 0.001) #implementato
    delta = 8  + 0*random.gauss(0.001, 0.001) #implementato
    tan_gamma_half = math.tan(math.radians(gamma / 2)) #implementato
    sin_delta = math.sin(math.radians(delta))
    sin_alfa = math.sin(math.radians(alfa)) #implementato
    txm = (360 / math.pi / 2) * 2 *xm / external_diameter #implementato, corretto il raggio
    
    # Creazione dell'utensile
    points = [
        App.Vector(-(tool_depth + ym+10) * tan_gamma_half, +5, +10), # aggiunti 2 mm per estrusione esterna
        App.Vector((tool_depth + ym+10) * tan_gamma_half, +5, +10),
        App.Vector(0, +5 , 0 - (tool_depth + ym))
    ]
    
    tool_face = Part.makePolygon(points + [points[0]])
    tool_solid = Part.Face(tool_face).extrude(App.Vector(0, -external_diameter*0.5-5, 0))
    #qui potrei aggiungere due millimetri sulla lunghezza, ma non cambia nulla
    # Raccordo sulla punta dell'utensile
    try:
        tool_solid = tool_solid.makeFillet(tool_radius, tool_solid.Edges)
    except:
        print(f"Errore nel raccordo dell'utensile {i}, procedo senza fillet.")
    
    # Posizionamento e rotazione dell'utensile
    angle = i * teeth_angle + txm
    x = external_diameter * 0.50 * math.sin(math.radians(angle)) #cambiato da 0.505
    y = external_diameter * 0.50 * math.cos(math.radians(angle))
    z = h_base + h_master + crown_height
    
    # Rotazione attorno a Z per orientare il tool verso il centro
    rotation_z = App.Rotation(App.Vector(0, 0, 1), -angle)
    placement_1 = App.Placement(
    App.Vector(x, y, z),
    rotation_z 
    )
    rotation_z = App.Rotation(App.Vector(0, 0, 1), delta)
    rotation_x = App.Rotation(App.Vector(1, 0, 0), -alfa) 
    rotation_y = App.Rotation(App.Vector(0,1,0), teta)
    combined_rotation = rotation_z.multiply(rotation_x).multiply(rotation_y)  # Prima Z, poi Y locale
    placement_2 = App.Placement(
        App.Vector(0,0,0),
        combined_rotation
    )
    tool_solid.Placement = placement_1.multiply(placement_2)
    
    # Debug: Visualizza l'utensile prima del taglio
    tool_part = doc.addObject("Part::Feature", f"Tool_{i}")
    tool_part.Shape = tool_solid
    
    # Esegui l'operazione di sottrazione (cut)
    crown_with_teeth = crown_with_teeth.cut(tool_solid)
    
    print(f"Posizione utensile {i}: ({x}, {y}, {h_base + h_master + crown_height}), Angolo: {angle}")
 
# Aggiunta della corona dentata con i denti tagliati
final_crown = doc.addObject("Part::Feature", "CrownWithTeeth")
final_crown.Shape = crown_with_teeth
 
# Pulizia degli oggetti temporanei
for obj in doc.Objects:
    if obj.Name != "CrownWithTeeth":
        doc.removeObject(obj.Name)
 
doc.recompute()
doc.saveAs("GiranteDentata.FCStd")

 
print("Modellazione completata con errori e raccordo!")