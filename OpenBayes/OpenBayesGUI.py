###############################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network library
## Copyright (C) 2006  Gaitanis Kosta, Ronald Moncarey
##
## This library is free software; you can redistribute it and/or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the 
## Free Software Foundation, 
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
###############################################################################

"""
An OpenBayes GUI for the OpenBayes program created by Kosta Gaitanis & Elliot Cohen
OpenBayes GUI allow you to create/open Bayesian networks and saving them to XBN file format
"""

__version__ = '0.1'
__author__ = 'Ronald Moncarey'
__author_email__ = 'rmoncarey@gmail.com'

# !/usr/bin/python
# -*- coding: utf-8 -*-

import wx, os, sys, string
import wx.grid
import wx.lib.ogl as ogl
import OpenBayesXBN as obx

#Constantes menu Fichier
ID_NEW      = wx.NewId()
ID_OPEN       = wx.NewId()
ID_SAVE       = wx.NewId()
ID_SAVEAS   = wx.NewId()
ID_EXIT      = wx.NewId()

#Constantes menu Aide
ID_HELP        = wx.NewId()

#Liste de formes pour le creation des liens
shapes = []
#creations des variables globales
#qui vont contenir toute la structure du reseaux
bnInfos = obx.BnInfos()
listStatProp = obx.StaticProperties()
listDynProp = obx.DynamicProperties()
variables = obx.Variables()
listVar = []
listStruct = []
distributions = obx.Distribution()
listDistrib = []
#indique si une modification a ete faite sur le diagramme
modif = False
#auto-incremente le nom d'un nouveau noeud
nbNode = 0

#----------------------------------------------------------------------
#classe permettant le creation d'une forme en diaman
class DiamondShape(ogl.PolygonShape):
    def __init__(self, w, h):
        ogl.PolygonShape.__init__(self)
        points = [ (0.0,    -h/2.0),
                   (w/2.0,  0.0),
                   (0.0,    h/2.0),
                   (-w/2.0, 0.0),
                   ]

        self.Create(points)

#----------------------------------------------------------------------
#gestionnaire d'evenements des formes
class MyEvtHandler(ogl.ShapeEvtHandler):
    def __init__(self, frame, TW):
        ogl.ShapeEvtHandler.__init__(self)
        self.statbarFrame = frame
        self.win = frame
        self.TeWi = TW

#indique les coordonnes x y du noeud dans la barre de statut
    def UpdateStatusBar(self, shape):
        x, y = shape.GetX(), shape.GetY()
        width, height = shape.GetBoundingBoxMax()
        self.statbarFrame.SetStatusText("Pos: (%d, %d)  Size: (%d, %d)" %
                                        (x, y, width, height))
 
 #actualise la scrollbar afin qu'on puisse voir tout les noeuds       
    def UpdateScrollBar(self):
        shape = self.GetShape()
        canvas = shape.GetCanvas()
        shapeList = canvas.GetDiagram().GetShapeList()
        maxX = 0
        maxY = 0
        for s in shapeList:
            if s.GetX() > maxX:
                maxX = s.GetX()
            if s.GetY() > maxY:
                maxY = s.GetY()
        self.TeWi.SetScrollbars(20, 20, (maxX+50)/20, (maxY+50)/20)

#evenement gerant le clic gauche sur une forme
    def OnLeftClick(self, x, y, keys=0, attachment=0):    
        shape = self.GetShape()
        canvas = shape.GetCanvas()
        dc = wx.ClientDC(canvas)
        canvas.PrepareDC(dc)
        
        if shape.Selected():
            # Si l'objet est selectionne, on le deselectionne et on vide la liste des shapes
            del shapes[:]
            shape.Select(False, dc)
            #on redessine le diagramme pour supprimer l'image de la selection
            canvas.Redraw(dc)
        else:
            tb = self.win.GetToolBar()
            #si on est sur l'option pour tracer un lien
            if tb.GetToolState(60):
                shapes.append(shape)
                #si il y a au moins 2 shappes dans la liste
                #on trace le lien
                if len(shapes)>=2:
                    ShapesWindow.MyAddLine(self.TeWi, False)
            redraw = False
            shapeList = canvas.GetDiagram().GetShapeList()
            toUnselect = []

            #on deselection tous les shapes sauf celui sur lequel on a clique
            for s in shapeList:
                if s.Selected():
                    toUnselect.append(s)

            shape.Select(True, dc)

            if toUnselect:
                for s in toUnselect:
                    s.Select(False, dc)

                canvas.Redraw(dc)

        self.UpdateStatusBar(shape)
            
    def OnEndDragLeft(self, x, y, keys=0, attachment=0):
        shape = self.GetShape()
        ogl.ShapeEvtHandler.OnEndDragLeft(self, x, y, keys, attachment)

        if not shape.Selected():
            self.OnLeftClick(x, y, keys, attachment)

        self.UpdateStatusBar(shape)
        self.UpdateScrollBar()
        
        global modif
        modif = True

    def OnSizingEndDragLeft(self, pt, x, y, keys, attch):
        ogl.ShapeEvtHandler.OnSizingEndDragLeft(self, pt, x, y, keys, attch)
        self.UpdateStatusBar(self.GetShape())
        self.UpdateScrollBar()

    #si on fait un clic droit sur un shape
    #une fenetre modale s'ouvre avec les infos du noeud
    def OnRightClick(self, x, y, keys=0, attachment=0):
        shape = self.GetShape()
        canvas = shape.GetCanvas()
        dc = wx.ClientDC(canvas)
        canvas.PrepareDC(dc)
        
        if shape.Selected():
            pass
        else:
            shapeList = canvas.GetDiagram().GetShapeList()
            toUnselect = []

            for s in shapeList:
                if s.Selected():
                    toUnselect.append(s)
            shape.Select(True, dc)

            if toUnselect:
                for s in toUnselect:
                    s.Select(False, dc)

                canvas.Redraw(dc)
        dlg = ShapeDialog(shape)
        dlg.CenterOnScreen()
        dlg.ShowModal()
        dlg.Destroy()
        
        self.UpdateStatusBar(shape)

#----------------------------------------------------------------------
#construction de la grid customisable
#pour la fenetre Property Status
class CustomDataTablePS(wx.grid.PyGridTableBase):
    def __init__(self, vari):
        wx.grid.PyGridTableBase.__init__(self)

        self.data = []
        self.colLabels = ['VALUE']
        self.dataTypes = [wx.grid.GRID_VALUE_STRING]
        self.rowLabels = []
        
        #remplissage de la grid par les donnees
        x = 0
        for state in vari.stateName:
            self.data.append([''])
            self.rowLabels.append("State(" + str(x) + ")")
            self.SetValue(x, 0, state)
            x += 1
        self.rowLabels.append("State(" + str(x) + ")")
        
    def GetNumberRows(self):
        return len(self.data) + 1

    def GetNumberCols(self):
        return 1

    def IsEmptyCell(self, row, col):
        try:
            return not self.data[row][col]
        except IndexError:
            return True

    def GetValue(self, row, col):
        try:
            return self.data[row][col]
        except IndexError:
            return ''

    #fonction qui permet de rendre la grid dynamique
    #creation de cellule a la demande
    def SetValue(self, row, col, value):
        if (value == ""):
            self.RemoveData(row)
        
        else:
            try:
                self.data[row][col] = value
    
            except IndexError:
    
                # add a new row
                self.data.append([''])
                self.rowLabels.append("State(" + str(row+1) + ")")
                self.SetValue(row, col, value)
    
                # tell the grid we've added a row
                msg = wx.grid.GridTableMessage(self,
                        wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)
                
                self.GetView().ProcessTableMessage(msg)
    
    def RemoveData(self, row):
        
        self.data.pop(row)
        self.rowLabels.pop()
        msg = wx.grid.GridTableMessage(self,
                   wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED, row, 1)
        self.GetView().ProcessTableMessage(msg)

    def GetRowLabelValue(self, row):
        try:
            return self.rowLabels[row]
        except:
            return ''
        
    def GetColLabelValue(self, col):
        return self.colLabels[col]

#construction de la grid customisable
#pour la fenetre des noms de distribution du noeud   
class CustomDataTablePDN(wx.grid.PyGridTableBase):
    def __init__(self, distri, vari):
        wx.grid.PyGridTableBase.__init__(self)

        self.dis = distri
        self.data = []
        self.colLabels = []
        self.dataTypes = [wx.grid.GRID_VALUE_STRING]
        self.rowLabels = []

        #remplissage des probabilites du noeud
        x = 0
        y = 0
        nbElem = len(self.dis.condelem)

        #si le noeud n'a pas de parents
        if nbElem == 0:
            self.colLabels.append(distri.name)
            for state in vari.stateName:
                self.SetValue(x, 0, state)
                x += 1
        #sinon
        else:
            #type Discrete
            if distri.type == "discrete":
                st = 0
                case = 1
                r = 1
                t = 1
                #inexplicable: 5h de reflexion pour y arriver ;)
                #on compte d'abbord le nombre de lignes a remplir
                for elem in distri.condelem:
                    for var in listVar:
                        if (var.name == elem):
                            for state in var.stateName:
                                st += 1
                    case *= st
                    st = 0
    
                #on remplit la grid avec les valeurs recuperees
                for elem in distri.condelem:
                    self.colLabels.append(elem)
                    for var in listVar:
                        if (var.name == elem):
                            test = len(var.stateName)
                            while r > 0:
                                for state in var.stateName:
                                    k = 0
                                    while k < case/test:
                                        self.SetValue(x, y, state)
                                        k += 1
                                        x += 1
                                r -= 1
                            break
                    case = case/test
                    x = 0
                    y += 1
                    t *= test
                    r = t
                    
            #type Causally Independent
            else:
                st = 0
                case = 1
                r = 1
                t = 1
                for elem in distri.condelem:
                    for var in listVar:
                        if (var.name == elem):
                            for state in var.stateName:
                                st += 1
                            st -= 1
                    case += st
                    st = 0
                
                x = 0
                ligne = 1
                premligne = True
                #on remplit la grid avec les valeurs recuperees
                for elem in distri.condelem:
                    self.colLabels.append(elem)
                    for var in listVar:
                        if (var.name == elem):
                            nbstate = len(var.stateName)-1
                            s = 1
                            while x < case:
                                if premligne:
                                    self.SetValue(x, y, "C.I.")
                                    premligne = False
                                elif (x >= ligne) and (s <= nbstate):
                                    self.SetValue(ligne, y, var.stateName[s])
                                    s += 1
                                    ligne += 1
                                else:
                                    self.SetValue(x, y, ("(" + var.stateName[0] + ")"))
                                x += 1
                    
                    premligne = True
                    x = 0
                    y += 1

                
    def GetNumberRows(self):
        return len(self.data)

    def GetNumberCols(self):
        if len(self.dis.condelem) == 0:
            return 1
        else:
            return len(self.dis.condelem)

    def IsEmptyCell(self, row, col):
        try:
            return not self.data[row][col]
        except IndexError:
            return True

    def GetValue(self, row, col):
        try:
            return self.data[row][col]
        except IndexError:
            return ''

    def SetValue(self, row, col, value):

        try:
            self.data[row][col] = value

        except IndexError:

            # add a new row
            self.data.append([''] * self.GetNumberCols())
            self.SetValue(row, col, value)

            # tell the grid we've added a row
            wx.grid.GridTableMessage(self,
                    wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)


    def GetRowLabelValue(self, row):
        try:
            return self.rowLabels[row]
        except:
            return ''
        
    def GetColLabelValue(self, col):
        return self.colLabels[col]
    
#construction de la grid customisable
#pour la fenetre des valeurs de distribution du noeud     
class CustomDataTablePDV(wx.grid.PyGridTableBase):
    def __init__(self, distri, vari):
        wx.grid.PyGridTableBase.__init__(self)

        self.dis = distri
        self.var = vari
        self.data = []
        self.colLabels = []
        self.dataTypes = [wx.grid.GRID_VALUE_STRING]
        self.rowLabels = []

        x = 0
        y = 0

        nbElem = len(self.dis.condelem)
        if nbElem == 0:
            self.colLabels.append(distri.name)

            for data in distri.dpiData:
                for val in string.split(data):
                    self.SetValue(x, 0, val)
                    x += 1

        else:
            for state in vari.stateName:
                self.colLabels.append(state)
                
            for data in distri.dpiData:
                for val in string.split(data):
                    self.SetValue(x, y, val)
                    y += 1
                y = 0
                x += 1 
        
    def GetNumberRows(self):
        return len(self.data)

    def GetNumberCols(self):
        if len(self.dis.condelem) == 0:
            return 1
        else:
            return len(self.var.stateName)

    def IsEmptyCell(self, row, col):
        try:
            return not self.data[row][col]
        except IndexError:
            return True

    def GetValue(self, row, col):
        try:
            return self.data[row][col]
        except IndexError:
            return ''

    def SetValue(self, row, col, value):

        try:
            self.data[row][col] = value

        except IndexError:

            # add a new row
            self.data.append([''] * self.GetNumberCols())
            self.SetValue(row, col, value)

            # tell the grid we've added a row
            wx.grid.GridTableMessage(self,
                    wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)


    def GetRowLabelValue(self, row):
        try:
            return self.rowLabels[row]
        except:
            return ''
        
    def GetColLabelValue(self, col):
        return self.colLabels[col]

#construction de la grid customisable
#pour la fenetre des Property du noeud         
class CustomDataTablePP(wx.grid.PyGridTableBase):
    def __init__(self, vari):
        wx.grid.PyGridTableBase.__init__(self)

        self.dataTypes = [wx.grid.GRID_VALUE_STRING
                          ]
        self.data = []
        self.colLabels = ['VALUE']
        self.rowLabels = []
        
        try:
            x = 0
            
            for p in listDynProp.dynPropType:
                #notes rajoutees pas MSBNx: "DTASDG_Notes" et "MS_Addins"
                #et qu'il ne faut pas afficher dans la grid
                if (p['NAME'] != "DTASDG_Notes") and (p['NAME'] != "MS_Addins"):
            
                    if (vari.propertyNameValue.has_key(p['NAME'])):
                        self.rowLabels.append(p['NAME'])
                        self.SetValue(x, 0, vari.propertyNameValue[p['NAME']])
                        x += 1
                    else:
                        self.rowLabels.append(p['NAME'])
                        self.SetValue(x, 0, '')
                        x += 1
        except:
            pass

    def GetNumberRows(self):
        return len(self.data)

    def GetNumberCols(self):
        return 1

    def IsEmptyCell(self, row, col):
        try:
            return not self.data[row][col]
        except IndexError:
            return True

    def GetValue(self, row, col):
        try:
            return self.data[row][col]
        except IndexError:
            return ''

    def SetValue(self, row, col, value):
        try:
            self.data[row][col] = value
        except IndexError:
            # add a new row
            self.data.append([''])
            self.SetValue(row, col, value)

            # tell the grid we've added a row
            wx.grid.GridTableMessage(self,
                    wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)

    def GetRowLabelValue(self, row):
        try:
            return self.rowLabels[row]
        except:
            return ''
        
    def GetColLabelValue(self, col):
        return self.colLabels[col]

#initialisation de la grid du Property State
class GridShapePS(wx.grid.Grid):
    def __init__(self, parent, vari):

        self.table = CustomDataTablePS(vari)
        parent.SetTable(self.table, True)
        parent.SetColLabelSize(0)
        parent.SetRowLabelSize(100)
        parent.AutoSizeColumns(False)
        
#initialisation de la grid Distribution (noms)
class GridShapePDN(wx.grid.Grid):
    def __init__(self, parent, distri, vari):

        self.table = CustomDataTablePDN(distri, vari)
        parent.SetTable(self.table, True)
        parent.SetColLabelSize(50)
        parent.SetRowLabelSize(0)
        parent.AutoSizeColumns(False)

#initialisation de la grid Distribution (valeurs)        
class GridShapePDV(wx.grid.Grid):
    def __init__(self, parent, distri, vari):

        self.table = CustomDataTablePDV(distri, vari)
        parent.SetTable(self.table, True)
        parent.SetColLabelSize(50)
        parent.SetRowLabelSize(0)
        parent.AutoSizeColumns(False)

#initialisation de la grid Property
class GridShapePP(wx.grid.Grid):
    def __init__(self, parent, vari):
        self.table = CustomDataTablePP(vari)
        parent.SetTable(self.table, True)
        parent.SetColSize(0, 300)
        parent.SetRowLabelSize(150)

#Initialisation du panel Property Node        
class ShapeDialogPN(wx.Panel):
    def __init__(self, parent, vari, shape):
        self.var = vari
        self.shap = shape
 
        wx.StaticText(parent, -1, "Name:", pos=(10,10))
        self.name = wx.TextCtrl(parent, -1, vari.name, pos=(200,10), size=(200,-1))
        wx.StaticText(parent, -1, "Description:", pos=(10,35))
        self.desc = wx.TextCtrl(parent, -1, vari.description, pos=(200,35), size=(200,-1))
        TypesList = ['discrete', 'continue']
        wx.StaticText(parent, -1, "Type:", pos=(10,60))
        self.type = wx.ComboBox(parent, -1, vari.type, (200, 60),
                         (200, -1), TypesList, wx.CB_DROPDOWN)
        
        b = wx.Button(parent, -1, "Apply", (520, 310))
        parent.Bind(wx.EVT_BUTTON, self.OnClick, b)
        b.SetDefault()
        b.SetSize(b.GetBestSize())
    
    #Action qui sauve les donnees de la grid
    #dans les variables globales les concernants
    def OnClick(self, event):
        oldName = self.shap.GetRegions()[0].GetFormattedText()[0].GetText()
        self.var.name = self.name.GetValue()
        self.var.description = self.desc.GetValue()
        self.var.type = self.type.GetValue()

        tempStruct = listStruct[:]
        del listStruct [:]
        for name in tempStruct:
            if (name == oldName):
                listStruct.append(self.var.name)
            else:
                listStruct.append(name)

        tempDist = listDistrib[:]
            
        for dis in listDistrib:
            if (dis.name == oldName):
                dis.name = self.var.name

            tempCondelem = dis.condelem[:]
            del dis.condelem[:]
            
            for elem in tempCondelem:
                if (elem == oldName):
                    dis.condelem.append(self.var.name)
                else:
                    dis.condelem.append(elem)
                
        self.shap.GetRegions()[0].GetFormattedText()[0].SetText(self.name.GetValue())
        canvas = self.shap.GetCanvas()
        dc = wx.ClientDC(canvas)
        canvas.PrepareDC(dc)
        canvas.Redraw(dc)
        
        global modif
        modif = True

#Initialisation de la grid Property State
class ShapeDialogPS(wx.grid.Grid):
    def __init__(self, parent, distri, vari):
        self.var = vari
        self.dis = distri
        self.gridShapePS = wx.grid.Grid(parent, -1, size = (650, 300))
        self.gps = GridShapePS(self.gridShapePS, vari)

        b = wx.Button(parent, -1, "Apply", (520, 310))
        parent.Bind(wx.EVT_BUTTON, self.OnClick, b)
        b.SetDefault()
        b.SetSize(b.GetBestSize())
    
    #Action qui sauve les donnees de la grid
    #dans les variables globales les concernants    
    def OnClick(self, event):
        self.var.stateName = []
        
        for x in range(self.gps.table.GetNumberRows()):

            if (self.gps.table.GetValue(x, 0) != ""):
                self.var.stateName.append(self.gps.table.GetValue(x, 0))
        y = self.gps.table.GetNumberRows()-1
        num = 1.0/y
        del self.dis.dpiData[:]
        stri = str(num) + " "
        
        for x in range(self.gps.table.GetNumberRows()-2):

            if (self.gps.table.GetValue(x, 0) != ""):
                stri += str(num) + " "
        self.dis.dpiData.append(stri)
        
        global modif
        modif = True

#Initialisation de la grid Distribution        
class ShapeDialogPD(wx.grid.Grid):
    def __init__(self, parent, distri, vari):
        self.dis = distri
        self.var = vari
        self.win = parent
        self.gridShapePDN = wx.grid.Grid(parent, -1, size = (350, 300), pos = (0, 0))
        self.gpdn = GridShapePDN(self.gridShapePDN, distri, vari)
        self.gridShapePDV = wx.grid.Grid(parent, -1, size = (290, 300), pos = (350,0))
        self.gpdv = GridShapePDV(self.gridShapePDV, distri, vari)

        b = wx.Button(parent, -1, "Apply", (520, 310))
        parent.Bind(wx.EVT_BUTTON, self.OnClick, b)
        b.SetDefault()
        b.SetSize(b.GetBestSize())
    
    #Action qui sauve les donnees de la grid
    #dans les variables globales les concernants    
    def OnClick(self, event):
        self.dis.dpiData = []
        val = ""
        
        if len(self.dis.condelem) == 0:
            
            for x in range(self.gpdv.table.GetNumberRows()):
                val += self.gpdv.table.GetValue(x, 0) + " "
            self.dis.dpiData.append(val)
        
        else:
                  
            for x in range(self.gpdv.table.GetNumberRows()):
                for y in range(self.gpdv.table.GetNumberCols()):
                    val += self.gpdv.table.GetValue(x, y) + " "
                self.dis.dpiData.append(val)
        
        global modif
        modif = True

#Initialisation de la grid Property    
class ShapeDialogPP(wx.grid.Grid):
    def __init__(self, parent, vari):
        self.var = vari
        self.win = parent
        self.gridShapePP = wx.grid.Grid(parent, -1, size = (650, 300))
        self.gpp = GridShapePP(self.gridShapePP, vari)

        b = wx.Button(parent, -1, "Apply", (520, 310))
        parent.Bind(wx.EVT_BUTTON, self.OnClick, b)
        b.SetDefault()
        b.SetSize(b.GetBestSize())
        
    #Action qui sauve les donnees de la grid
    #dans les variables globales les concernants
    def OnClick(self, event):
        self.var.propertyNameValue.clear()
        
        for x in range(self.gpp.table.GetNumberRows()):

            if (self.gpp.table.GetValue(x, 0) != ""):
                
                for p in listDynProp.dynPropType:
                    
                    if (p['NAME'] == self.gpp.table.GetRowLabelValue(x)):
                        TypesList = ['string', 'realarray', 'stringarray',
                     'enumeration']
                        
                        if (p['TYPE'] in TypesList):
                            self.var.propertyNameValue[self.gpp.table.GetRowLabelValue(x)] = self.gpp.table.GetValue(x, 0)
                            
                        else:
                            try:
                                int(self.gpp.table.GetValue(x, 0))
                                self.var.propertyNameValue[self.gpp.table.GetRowLabelValue(x)] = self.gpp.table.GetValue(x, 0)
                            except:
                                dlg = wx.MessageDialog(self.win, "Need a real",
                                                       "Error", wx.OK | wx.ICON_ERROR)
                                dlg.ShowModal()
                                dlg.Destroy()
        
        global modif
        modif = True

#Creation de la fenetre modale des infos du noeud
class ShapeDialog(wx.Dialog):
    def __init__(self, shape):
        wx.Dialog.__init__(self, None, -1, "Node infos", size = (650, 400),
             style = wx.DEFAULT_DIALOG_STYLE)
        self.tabs = wx.Notebook(self, -1, style = 0)
        self.panelNode = wx.Panel(self.tabs, -1)
        self.panelState = wx.Panel(self.tabs, -1)
        self.panelDistri = wx.Panel(self.tabs, -1)
        self.panelProp = wx.Panel(self.tabs, -1)
        self.panelPropType = wx.Panel(self.tabs, -1)
        
        #ajout des onglets sur la fenetre
        self.tabs.AddPage(self.panelNode, "Node", True)
        self.tabs.AddPage(self.panelState, "State", False)
        self.tabs.AddPage(self.panelDistri, "Distribution", False)
        self.tabs.AddPage(self.panelProp, "Property", False)
        self.tabs.AddPage(self.panelPropType, "Property Types", False)
        
        
        #bug d'affichage sous windows si on ne met pas ceci:
        self.ProcessEvent(wx.SizeEvent((-1,-1)))
        
        nodeName = shape.GetRegions()[0].GetFormattedText()[0].GetText()
        self.SetTitle(nodeName)
        
        x = shape.GetX()
        y = shape.GetY()
     
        for var in listVar:
            if (var.name == nodeName):
                vari = var       
                break
            
        for dis in listDistrib:
            if (dis.name == nodeName):
                distri = dis       
                break

        #fonctions qui vont initialiser le fenetre
        ShapeDialogPN(self.panelNode, vari, shape)
        ShapeDialogPS(self.panelState, distri, vari)
        ShapeDialogPD(self.panelDistri, distri, vari)
        ShapeDialogPP(self.panelProp, vari)
        DiagramDialogPPT(self.panelPropType)
        
#construction de la grid customisable
#pour la fenetre Diagram Types
class CustomDataTableTypes(wx.grid.PyGridTableBase):
    def __init__(self):
        wx.grid.PyGridTableBase.__init__(self)

        self.dataTypes = [wx.grid.GRID_VALUE_STRING
                          ]
        self.data = []
        self.rowLabels = []
        
        try:
            x = 0
             
            for p in listDynProp.dynPropType:
                #notes rajoutees pas MSBNx: "DTASDG_Notes" et "MS_Addins"
                #et qu'il ne faut pas afficher dans la grid
                if (p['NAME'] != "DTASDG_Notes") and (p['NAME'] != "MS_Addins"):
                    self.rowLabels.append('Name')
                    self.SetValue(x, 0, p['NAME'])
                    x += 1
                
                    self.rowLabels.append('Data Type')
                    self.SetValue(x, 0, p['TYPE'])
                    x += 1
                    
                    if (p.has_key('ENUMSET')):
                        self.rowLabels.append('Valid Values')
                        self.SetValue(x, 0, p['ENUMSET'])
                        x += 1
                    
                    if (p.has_key('COMMENT')):
                        self.rowLabels.append('Comment')
                        self.SetValue(x, 0, p['COMMENT'])
                        x += 1
                    
                    self.rowLabels.append('')
                    self.SetValue(x, 0, '')
                    x +=1
        except:
            pass
        

    def GetNumberRows(self):
        return len(self.data)

    def GetNumberCols(self):
        return 1

    def IsEmptyCell(self, row, col):
        try:
            return not self.data[row][col]
        except IndexError:
            return True

    def GetValue(self, row, col):
        try:
            return self.data[row][col]
        except IndexError:
            return ''

    def SetValue(self, row, col, value):
        try:
            self.data[row][col] = value
        except IndexError:
            # add a new row
            self.data.append([''])
            self.SetValue(row, col, value)

            # tell the grid we've added a row
            wx.grid.GridTableMessage(self,
                    wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)

    def GetRowLabelValue(self, row):
        try:
            return self.rowLabels[row]
        except:
            return ''

#construction de la grid customisable
#pour la fenetre Diagram Model
class CustomDataTableModel(wx.grid.PyGridTableBase):
    def __init__(self):
        wx.grid.PyGridTableBase.__init__(self)

        self.dataTypes = [wx.grid.GRID_VALUE_STRING
                          ]
        self.data = []
        self.colLabels = ['VALUE']
        self.rowLabels = []
        
        try:
            x = 0
            
            for p in listDynProp.dynPropType:
                #notes rajoutees pas MSBNx: "DTASDG_Notes" et "MS_Addins"
                #et qu'il ne faut pas afficher dans la grid
                if (p['NAME'] != "DTASDG_Notes") and (p['NAME'] != "MS_Addins"):
            
                    if (listDynProp.dynProperty.has_key(p['NAME'])):
                        self.rowLabels.append(p['NAME'])
                        self.SetValue(x, 0, listDynProp.dynProperty[p['NAME']])
                        x += 1
                    else:
                        self.rowLabels.append(p['NAME'])
                        self.SetValue(x, 0, '')
                        x += 1
        except:
            pass
        

    def GetNumberRows(self):
        return len(self.data)

    def GetNumberCols(self):
        return 1

    def IsEmptyCell(self, row, col):
        try:
            return not self.data[row][col]
        except IndexError:
            return True

    def GetValue(self, row, col):
        try:
            return self.data[row][col]
        except IndexError:
            return ''

    def SetValue(self, row, col, value):
        try:
            self.data[row][col] = value
        except IndexError:
            # add a new row
            self.data.append([''])
            self.SetValue(row, col, value)

            # tell the grid we've added a row
            wx.grid.GridTableMessage(self,
                    wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)

    def GetRowLabelValue(self, row):
        try:
            return self.rowLabels[row]
        except:
            return ''
    def GetColLabelValue(self, col):
        return self.colLabels[col]

#initialisation de la grid Node Types    
class GridPropTypes(wx.grid.Grid):
    def __init__(self, parent):

        table = CustomDataTableTypes()
        parent.SetTable(table, True)
        parent.SetColLabelSize(0)
        parent.SetRowLabelSize(100)
        parent.AutoSizeColumns(False)

#initialisation de la grid Node Model
class GridPropModel(wx.grid.Grid):
    def __init__(self, parent):

        self.table = CustomDataTableModel()
        parent.SetTable(self.table, True)
        parent.SetColSize(0, 300)
        parent.SetRowLabelSize(150)

#Creation de la fenetre modale qui permet d'ajouter un type  
class AddType(wx.Dialog):
    name = None
    type = None
    comment = None
    enumset = None
    
    def __init__(self):
        wx.Dialog.__init__(self, None, -1, "Add Type", size = (500, 300),
             style = wx.DEFAULT_DIALOG_STYLE)
        text = wx.StaticText(self, -1, "Name", pos=(20,10))
        text.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.name = wx.TextCtrl(self, -1, "", pos = (100,10), size = (200,-1))
        text = wx.StaticText(self, -1, "Type", pos=(20,45))
        text.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL))
        
        #la liste des types autorises par xbn
        TypesList = ['real', 'string', 'realarray', 'stringarray',
                     'enumeration']
        self.type = wx.ComboBox(self, -1, "", (100, 45),
                         (200, -1), TypesList, wx.CB_DROPDOWN)
        
        text = wx.StaticText(self, -1, "Comment", pos=(10,80))
        text.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.comment = wx.TextCtrl(self, -1, "", pos = (10,100), size = (450,-1))
        text = wx.StaticText(self, -1, "Enumerated Values (separete with space)", pos=(10,135))
        text.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.enumset = wx.TextCtrl(self, -1, "", pos = (10,155), size = (450,50), style = wx.TE_MULTILINE)
        okBut = wx.Button(self, wx.ID_OK, "OK", pos = (400,240))
        CancelBut = wx.Button(self, wx.ID_CANCEL, "Cancel", pos = (300,240))

#Initialisation du panel diagram Property Types    
class DiagramDialogPPT(wx.Panel):
    def __init__(self, parent):
        self.win = parent
        self.gridPropType = wx.grid.Grid(parent, -1, size = (650, 300))
        GridPropTypes(self.gridPropType)

        b = wx.Button(parent, -1, "Add Type", (20, 310))
        parent.Bind(wx.EVT_BUTTON, self.OnClick, b)
        b.SetDefault()
        b.SetSize(b.GetBestSize())

    #Action qui sauve les donnees de la grid
    #dans les variables globales les concernants
    def OnClick(self, event):
        dlg = AddType()
        dlg.CenterOnScreen()
        result = dlg.ShowModal()
        if result == wx.ID_OK:
            dictDyn = {}
            if (dlg.name.GetValue() != "") and (dlg.type.GetValue() != ""):
                dictDyn["NAME"] = dlg.name.GetValue()
                dictDyn["TYPE"] = dlg.type.GetValue()
                if (dlg.comment.GetValue() != ""):
                    dictDyn["COMMENT"] = dlg.comment.GetValue()
                if (dlg.type.GetValue() == "enumeration"):
                    dictDyn["ENUMSET"] = dlg.enumset.GetValue()
            
                listDynProp.dynPropType.append(dictDyn)

            #fait un refresh de la grid pour afficher les nouveaux types
            GridPropTypes(self.gridPropType)
            
        dlg.Destroy()
        
        global modif
        modif = True

#Initialisation du panel diagram Property Model
class DiagramDialogPM(wx.Panel):
    def __init__(self, parent):
        self.gridPM = wx.grid.Grid(parent, -1, size = (650, 300))
        self.gpm = GridPropModel(self.gridPM)

        b = wx.Button(parent, -1, "Apply", (520, 310))
        parent.Bind(wx.EVT_BUTTON, self.OnClick, b)
        b.SetDefault()
        b.SetSize(b.GetBestSize())
        
    #Action qui sauve les donnees de la grid
    #dans les variables globales les concernants
    def OnClick(self, event):
        listDynProp.dynProperty.clear()
        
        for x in range(self.gpm.table.GetNumberRows()):

            if (self.gpm.table.GetValue(x, 0) != ""):
                listDynProp.dynProperty[self.gpm.table.GetRowLabelValue(x)] = self.gpm.table.GetValue(x, 0)
        
        global modif
        modif = True        

#Initialisation du panel diagram Infos      
class DiagramInfos(wx.Panel):
    name = None
    
    def __init__(self, parent):

        if (bnInfos.root != None):
            root = bnInfos.root
        else:
            root = "bndefault"
       
        if (listStatProp.format != None):
            format = listStatProp.format
            version = listStatProp.version
            creator = listStatProp.creator
        else:
            format = "MSR DTAS XML"
            version = "1.0"
            creator = "Open Bayes GUI"
        
        text = wx.StaticText(parent, -1, "Name:", pos=(10,30))
        text.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.name = wx.TextCtrl(parent, -1, root, pos=(200,30), size=(200,-1))
        text = wx.StaticText(parent, -1, "XBN_Format:", pos=(10,60))
        text.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
        wx.StaticText(parent, -1, format, pos=(200,60), size=(200,-1))
        text = wx.StaticText(parent, -1, "XBN_Version:", pos=(10,90))
        text.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
        wx.StaticText(parent, -1, version, pos=(200,90), size=(200,-1))
        text = wx.StaticText(parent, -1, "XBN_Creator:", pos=(10,120))
        text.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
        wx.StaticText(parent, -1, creator, pos=(200,120), size=(200,-1))

        b = wx.Button(parent, -1, "Apply", (520, 310))
        parent.Bind(wx.EVT_BUTTON, self.OnClick, b)
        b.SetDefault()
        b.SetSize(b.GetBestSize())
        
    #Action qui sauve les donnees du panel
    #dans les variables globales les concernants
    def OnClick(self, event):
        bnInfos.root = self.name.GetValue()
        
        global modif
        modif = True

#Creation de la fenetre modale des infos du diagram        
class DiagramDialog(wx.Dialog):
    def __init__(self):
        wx.Dialog.__init__(self, None, -1, "Diagram properties", size = (650, 400),
             style = wx.DEFAULT_DIALOG_STYLE)
        self.tabs = wx.Notebook(self, -1, style = 0)
        self.panelDiagramInfos = wx.Panel(self.tabs, -1)
        self.panelPropType = wx.Panel(self.tabs, -1)
        self.panelModel = wx.Panel(self.tabs, -1)
        
        #Creation des onglets
        self.tabs.AddPage(self.panelDiagramInfos, "Diagram Infos", True)
        self.tabs.AddPage(self.panelPropType, "Property Types", False)
        self.tabs.AddPage(self.panelModel, "Model", False)
        
        #bug d'affichage sous windows si on ne met pas ceci:
        self.ProcessEvent(wx.SizeEvent((-1,-1)))

        DiagramInfos(self.panelDiagramInfos)
        DiagramDialogPPT(self.panelPropType)
        DiagramDialogPM(self.panelModel)
        
#----------------------------------------------------------------------
#Fenetre qui contient le diagramme afin d'afficher les formes
class ShapesWindow(ogl.ShapeCanvas):
    def __init__(self, parent):
        ogl.ShapeCanvas.__init__(self, parent)

        self.frame = parent
        self.SetBackgroundColour("WHITE")
        self.diagram = ogl.Diagram()
        self.SetDiagram(self.diagram)
        self.diagram.SetCanvas(self)

        dc = wx.ClientDC(self)
        self.PrepareDC(dc)

    #Fonction qui ajoute une forme sur le diagramme
    def MyAddShape(self, shape, x, y, pen, brush, text, fileOpen):
        #initialisation de la forme
        shape.SetDraggable(True, True)
        shape.SetCanvas(self)
        shape.SetX(x)
        shape.SetY(y)
        if pen:    shape.SetPen(pen)
        if brush:  shape.SetBrush(brush)
        if text:
            for line in text.split('\n'):
                shape.AddText(line)

        self.diagram.AddShape(shape)
        shape.Show(True)

        #ajout de la gestion d'evenement sur la forme
        evthandler = MyEvtHandler(self.frame, self)
        evthandler.SetShape(shape)
        evthandler.SetPreviousHandler(shape.GetEventHandler())
        shape.SetEventHandler(evthandler)
        
        #si on est en mode d'ouverture de fichier
        #on ne doit pas initialiser les noeuds qui s'affichent
        if not fileOpen:
            dc = wx.ClientDC(self)
            self.PrepareDC(dc)
            self.Redraw(dc)
            
            #initialisation des infos du noeud
            global nbNode
            global variables    
            variables = obx.Variables()
            variables.name = "node_" + str(nbNode)
            variables.description = ""
            variables.type = "discrete"
            variables.xpos = str(x)
            variables.ypos = str(y)
            variables.stateName = ["Yes", "No"]
            #ajout du noeud aux variables globales
            listVar.append(variables)
            
            #initialisation de la distribution
            #des probabilites du noeud
            global distributions
            distributions = obx.Distribution()
            distributions.name = "node_" + str(nbNode)
            distributions.type = "discrete"
            distributions.condelem = []
            distributions.dpiIndex = []
            distributions.dpiData = ["0.5 0.5 "]
            #ajout de la distribution aux variables globales
            listDistrib.append(distributions)
            
            global modif
            modif = True
            
            #on incremente le compteur de noeud
            nbNode += 1
            
        return shape
    
    #ToDo
    def MyDelShape(self):
        pass
    
    #Fonction qui permet de tracer un lien entre 2 formes
    def MyAddLine(self, fileOpen):
        fromShape = shapes[0]
        toShape = shapes[1]
        
        shapes[0] = shapes[1]
        del shapes[1]
        #initialisation du lien
        line = ogl.LineShape()
        line.SetCanvas(self)
        line.SetPen(wx.BLACK_PEN)
        line.SetBrush(wx.BLACK_BRUSH)
        line.AddArrow(ogl.ARROW_ARROW)
        line.MakeLineControlPoints(2)
        fromShape.AddLine(line, toShape)
        self.diagram.AddShape(line)
        line.Show(True)
        
        #si on est en mode d'ouverture de fichier
        #on ne doit pas initialiser les liens qui s'affichent
        if not fileOpen:
            #ajout du lien aux variables globales
            listStruct.append(fromShape.GetRegions()[0].GetFormattedText()[0].GetText())
            listStruct.append(toShape.GetRegions()[0].GetFormattedText()[0].GetText())
            

            for dist in listDistrib:
                if (dist.name == toShape.GetRegions()[0].GetFormattedText()[0].GetText()):

                    dist.condelem.append(fromShape.GetRegions()[0].GetFormattedText()[0].GetText())
                    
                    del dist.dpiIndex[:]
                    del dist.dpiData[:]
                    
                    #initialise l'index des distributions
                    #Ne fait pas ce qu'il faut mais cela fonctionne
                    #aussi bien sous OpenBayes GUI que sous MSBNx
                    st = 0
                    case = 1
                    for elem in dist.condelem:
                        for var in listVar:
                            if (var.name == elem):
                                for state in var.stateName:
                                    st += 1
                        case *= st
                        st = 0
                
                    for state in listVar:
                        if (state.name == dist.name):
                            nbState = len(state.stateName)

                    x = 0
                    y = 0
                    val = ""

                    while case > 0:
                        temp = nbState
                        while temp > 0:
                            val += str(x) + " "
                            temp -= 1
                        dist.dpiIndex.append(val)
                        val = ""
                        dist.dpiData.append("0.5 0.5 ")
                        case -= 1
            
            global modif
            modif = True
                        
    #gestin d'evenement qui recupere un clic gauche sur le diagramme
    def OnLeftClic (self, event):
        tb = self.frame.GetToolBar()
        if tb.GetToolState(10):
            pass
        #si l'icone est selectionnee alors on ajoute un noeud
        elif tb.GetToolState(20):
            point = event.GetPosition()
            self.MyAddShape(ogl.EllipseShape(75, 50), 
                            point.x, point.y, wx.Pen("BLACK", 3), wx.Brush("#00BFFF"), "node_" + str(nbNode), False
                            )
        #je ne prend pas en compte la creation
        #d'autre type de noeuds pour l'instant
        elif tb.GetToolState(30):
            point = event.GetPosition()
            self.MyAddShape(ogl.CircleShape(30), 
                            point.x, point.y, wx.Pen("BLACK", 3), wx.Brush("#1c1cc4"), "constr.", False
                            )
        #idem
        elif tb.GetToolState(40):
            point = event.GetPosition()
            self.MyAddShape(DiamondShape(50, 50), 
                            point.x, point.y, wx.Pen("BLACK", 3, wx.DOT), wx.Brush("#ff6666"), "use", False
                            )
        #idem
        elif tb.GetToolState(50):
            point = event.GetPosition()
            self.MyAddShape(ogl.RectangleShape(50, 50), 
                            point.x, point.y, wx.Pen("BLACK", 3), wx.Brush("#60d660"), "decision", False
                            )
        elif tb.GetToolState(60):
            pass

#----------------------------------------------------------------------
#Affiche un splashscreen au lancement de l'application
class MySplashScreen(wx.SplashScreen):
    def __init__(self):
        bmp = wx.Image(opj("TBayes.png")).ConvertToBitmap()
        wx.SplashScreen.__init__(self, bmp,
                                 wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_TIMEOUT,
                                 2000, None, -1)
        self.Bind(wx.EVT_CLOSE, self.OnClose)


    def OnClose(self, event):
        event.Skip()
        self.Hide()

#Classe pour ouvrir un fichier
class OpenXbn:
    def __init__(self, parent, chemin):

        #on force les variables a etre globales
        global bnInfos
        global listStatProp
        global listDynProp
        global listVar
        global listStruct
        global listDistrib

        #Initialisation des variables globales
        xbn = obx.XmlParse(chemin)
        bnInfos = xbn.getBnInfos()
        listStatProp = xbn.getStaticProperties()
        listDynProp = xbn.getDynamicProperties()
        listVar = xbn.getVariablesXbn()
        listStruct = xbn.getStructureXbn()
        listDistrib = xbn.getDistribution()
        

        #Affichage des noeuds sur le diagramme
        #si le fichier ouvert est un xbn 0.2 on lance la procedure de conversion
        if xbn.version == "0.2":
            dlg = wx.MessageDialog(None,
       "Do you want to convert your v0.2 xbn file to a v1.0 xbn file ?",
                                   "Converter",
                                   wx.YES_NO | wx.ICON_QUESTION)
            result = dlg.ShowModal()
            if result == wx.ID_YES:
                #appel de la fonction de sauvegarde
                SaveXbn (chemin)
                #une fois sauve on reinitialise les variables
                parent.diagram.DeleteAllShapes()
                del shapes[:]
                parent.Refresh()
                bnInfos = None
                del listVar[:]
                del listStruct[:]
                del listDistrib[:]
                
                #appel de la fonction d'ouverture
                OpenXbn (parent, chemin)
            
            else:
                #appel de la fonction ouverture d'apres la version du xbn
                self.ShowShapes(parent, "0.2")
             
            dlg.Destroy()
            
        else:
            #appel de la fonction ouverture d'apres la version du xbn
            self.ShowShapes(parent, "1.0")
        
        parent.SetTitle(chemin)

        dc = wx.ClientDC(parent)
        parent.PrepareDC(dc)
        parent.Redraw(dc)

    #fonction d'affichage des noeuds
    def ShowShapes(self, parent, version):
        #Dictionary variables_name : shapes
        shapesLink = {}
        maxX = 0
        maxY = 0
        #les coordonnees x y des noeuds sont differentes
        #entre la versin 0.2 et 1.0
        #alors j'ai cree une formule qui permet d'afficher au mieux
        #l'un ou l'autre version
        for var in listVar:
            if (version == "0.2"):
                x = ((int(var.xpos))/10)-800
                y = ((int(var.ypos))/10)-800
            else:
                x = ((int(var.xpos))%1000)
                y = ((int(var.ypos))%1000)
            if x > maxX:
                maxX = x
            if y > maxY:
                maxY = y
            
            #Affiche la forme sur le diagramme
            shape = parent.MyAddShape(ogl.EllipseShape(75, 50), 
                x, y,
                    wx.Pen("BLACK", 3), wx.Brush("#00BFFF"),
                    var.name, True)
            shapesLink[var.name] = shape
            
        #Creation des liens entre les noeuds
        for struct in listStruct:
            shapes.append(shapesLink[struct])
            
            if len(shapes)>=2:
                ShapesWindow.MyAddLine(parent, True)
                del shapes[:]
                    
        #update de la scrollbar en fonction de la position des noeuds
        parent.SetScrollbars(20, 20, (maxX+50)/20, (maxY+50)/20)
        
#Application principale 
class Application(wx.App):
    def __init__(self):
        wx.App.__init__(self, redirect=False)

    def OnInit(self):
        #Lancement du splashscreen
        splash = MySplashScreen() 

        frame = wx.Frame(None, -1, "Open Bayes", pos=(0,0),
                        style=wx.DEFAULT_FRAME_STYLE)
        #creation d'une bar de status
        frame.CreateStatusBar()
        
        #initialisation du menu
        menuBar = wx.MenuBar()
        filemenu = wx.Menu()
        self.dest = None

        filemenu.Append(ID_NEW,"&New","Create file")
        filemenu.Append(ID_OPEN,"&Open","Open file")
        filemenu.Append(ID_SAVE,"&Save","Save file")
        filemenu.Append(ID_SAVEAS,"S&ave as","Save to another name")
        filemenu.Append(ID_EXIT,"&Quit","Quit program")
        
        self.Bind(wx.EVT_MENU, self.New, id=ID_NEW)
        self.Bind(wx.EVT_MENU, self.Open, id=ID_OPEN)
        self.Bind(wx.EVT_MENU, self.Save, id=ID_SAVE)
        self.Bind(wx.EVT_MENU, self.Saveas, id=ID_SAVEAS)
        self.Bind(wx.EVT_MENU, self.OnExitApp, id=ID_EXIT)
        menuBar.Append(filemenu, "&File")
         
        helpmenu = wx.Menu()
        helpmenu.Append(ID_HELP,"&Help","Print help")
        
        menuBar.Append(helpmenu, "&Help")
        
        #creation du menu
        frame.SetMenuBar(menuBar)
        frame.Show(True)
        #creation des gestionnaires d'evenements sur la fermeture de la frame
        frame.Bind(wx.EVT_CLOSE, self.OnCloseFrame)
        
        #variable qui servira a retirer ou ajouter une gestion
        #d'evenement sur un objet
        self.unbindclic = True
        #creation de la barre d'outils
        self.CreateTB(frame)
        
        #Creation des gestionnaires d'evenements de la barre d'outils
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=10)
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=20)
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=30)
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=40)
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=50)
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=60)
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=70)
        frame.Bind(wx.EVT_TOOL, self.OnToolClick, id=80)
        
        #initialisation des outils OGL
        ogl.OGLInitialize()
        win = ShapesWindow(frame)

        #on prend le handle de la fenetre qui contient le diagramme
        if win:
            frame.SetSize(size=wx.DisplaySize())
            win.SetFocus()
            self.window = win
            frect = frame.GetRect()

        else:
            frame.Destroy()
            return True

        self.SetTopWindow(frame)
        self.frame = frame

        return True
    
    #Fonction qui va creer la barre d'outils
    def CreateTB(self, frame):
        tb = frame.CreateToolBar( wx.TB_HORIZONTAL
                                 | wx.NO_BORDER
                                 | wx.TB_FLAT
                                 | wx.TB_TEXT
                                 )
        new_bmp =  wx.ArtProvider.GetBitmap(wx.ART_NEW, wx.ART_TOOLBAR)
        open_bmp = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR)
        save_bmp = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR)
        
        tb.AddSimpleTool(ID_NEW, new_bmp, "New")
        tb.AddSimpleTool(ID_OPEN, open_bmp, "Open")
        tb.AddSimpleTool(ID_SAVE, save_bmp, "Save")
        
        tb.AddSeparator()
        
        tb.AddCheckTool(10, wx.Image(opj("icones/move.png")).ConvertToBitmap(),
                               shortHelp="Move node")
        tb.AddCheckTool(20, wx.Image(opj("icones/node.png")).ConvertToBitmap(),
                               shortHelp="Create node")
        #tb.AddCheckTool(30, wx.Image(opj("icones/constraint.png")).ConvertToBitmap(),
        #                       shortHelp="Create constraint node")
        #tb.AddCheckTool(40, wx.Image(opj("icones/use.png")).ConvertToBitmap(),
        #                       shortHelp="Create use node")
        #tb.AddCheckTool(50, wx.Image(opj("icones/deci.png")).ConvertToBitmap(),
        #                       shortHelp="Create decision node")
        tb.AddCheckTool(60, wx.Image(opj("icones/link.png")).ConvertToBitmap(),
                               shortHelp="Create link")
        tb.AddCheckTool(70, wx.Image(opj("icones/delete.png")).ConvertToBitmap(),
                               shortHelp="Delete shape/link")
        tb.AddSeparator()
        
        tb.AddSimpleTool(80, wx.Image(opj("icones/diagram.png")).ConvertToBitmap(),
                               shortHelpString="Diagram properties")
        tb.ToggleTool(10,True)
        
        tb.Realize()
    
    #Fonction qui test si le diagramme a ete modifie
    def New(self, event):
        global modif

        #le diagramme a ete modifie ?
        if modif:
            #si oui on propose de le sauver
            dlg = wx.MessageDialog(self.window, "Want to save before new ?", "New",
                                   wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION)
            
            result = dlg.ShowModal()
            if result == wx.ID_YES:
                self.Save(event)

                if not modif:
                    self.Clear()
                
            elif result == wx.ID_NO:
                modif = False

                self.Clear()

            dlg.Destroy()
        
        else:
            self.Clear()
        
    #Fonction qui reinitialise le diagramme et toutes les variables globales
    def Clear(self):
        global nbNode
        nbNode = 0
        self.dest = None
        self.window.diagram.DeleteAllShapes()
        del shapes[:]
        self.window.Refresh()
 
        bnInfos.root = "bndefault"
        listStatProp.format = "MSR DTAS XML"
        listStatProp.version = "1.0"
        listStatProp.creator = "Open Bayes GUI"

        listDynProp.dynPropType = []
        listDynProp.dynProperty = {}
        listDynProp.dynPropXml = {}
        del listVar[:]
        del listStruct[:]
        del listDistrib[:]
            
    #Fonction qui teste si le fichier xbn est bon
    def Open(self, event):
        
        dlg = wx.FileDialog(self.window,"Filename","","","*.xbn",wx.OPEN)
        retour = dlg.ShowModal()
        
        if retour == wx.ID_OK:
            self.New(event)
            chemin = dlg.GetPath()
            
            try:
                OpenXbn(self.window, chemin)
                self.dest = chemin
                
            except:
                dlg = wx.MessageDialog(None,
                                       "You have selected a wrong XBN file",
                                       "Message", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()

    #Fonction qui teste si on a deja un chemin vers un fichier sauve
    def Save(self, event):
        if (self.dest == None):
            self.Saveas(event)
            
        else:
            SaveXbn(self.dest)
            global modif
            modif = False
    
    #Fonction qui va sauver le fichier xbn        
    def Saveas(self, event):
        
        dlg = wx.FileDialog(self.window,"Filename","","","*.xbn",wx.SAVE | wx.OVERWRITE_PROMPT)
        retour = dlg.ShowModal()
        
        if retour == wx.ID_OK:
            fichier = dlg.GetFilename()

            #on rajoute l'extension .xbn si ca n'a pas ete fait
            if (os.path.splitext(fichier)[1] != ".xbn"):
                fichier += ".xbn"
                #formatage pour Linux/Mac
                if dlg.GetDirectory().startswith('/'):
                    self.dest = dlg.GetDirectory() + '/' + fichier
                #formatage pour Windows
                else:
                    self.dest = dlg.GetDirectory() + '\\' + fichier
            else:
                self.dest = dlg.GetPath()
                
            SaveXbn(self.dest)
        
            global modif
            modif = False

        
    #gestion d'evenement pour la sortie de l'application
    def OnExitApp(self, event):
        self.frame.Close(True)

    #gestion d'evenement pour la fermeture de l'application
    def OnCloseFrame(self, event):
        if modif:
            dlg = wx.MessageDialog(self.window, "Want exit without save ?", "Exit",
                                   wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_NO:
                self.Saveas(event)
                if not modif:
                    event.Skip()
            else:
                event.Skip()
                
            dlg.Destroy()
        
        else:
            dlg = wx.MessageDialog(self.window, "Want to exit ?", "Exit",
                                   wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES:
                event.Skip()
            dlg.Destroy()
    
    #gestion d'evenement sur la barre d'outils
    def OnToolClick(self, event):
        tb = self.frame.GetToolBar()
        for n in (10,20,30,40,50,60,70):
            tb.ToggleTool(n,False)
        tb.ToggleTool(event.GetId(),True)
        if event.GetId()==10:
            del shapes[:]
            if not self.unbindclic:
                #en mode selection on desactive la gestion d'evenements
                #de la fenetre
                self.unbindclic = self.window.Unbind(wx.EVT_LEFT_DOWN, id=-1)
        elif event.GetId()==20:
            del shapes[:]
            if self.unbindclic:
                #en mode creation on active la gestion d'evenements
                #de la fenetre
                self.window.Bind(wx.EVT_LEFT_DOWN,
                                 self.window.OnLeftClic, id=-1)
                self.unbindclic = False
        elif event.GetId()==60:
            del shapes[:]
            if not self.unbindclic:
                #en mode creation de lien on desactive la gestion d'evenements
                #de la fenetre
                self.unbindclic = self.window.Unbind(wx.EVT_LEFT_DOWN, id=-1)
        elif event.GetId()==80:
            dlg = DiagramDialog()
            dlg.CenterOnScreen()
            dlg.ShowModal()
            dlg.Destroy()

#----------------------------------------------------------------------

def main():

    try:
        Path = os.path.dirname(__file__)
        os.chdir(Path)
    except:
        pass
    monappli = Application()
    monappli.MainLoop()
    
def opj(path):
    """Convert paths to the platform-specific separator"""
    str = apply(os.path.join, tuple(path.split('/')))
    # HACK: on Linux, a leading / gets lost...
    if path.startswith('/'):
        str = '/' + str
    return str
    
#----------------------------------------------------------------------

if __name__ == '__main__':
    main()

