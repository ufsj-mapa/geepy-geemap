import wx
import wx.html2
import tempfile
import os
import folium
import ee

ee.Initialize()

class mapBrowser(wx.Frame): 
    def __init__(self, *args, **kwds):
        """
        Initialize map window object.
        """ 
        wx.Frame.__init__(self, *args, **kwds) 
        sizer = wx.BoxSizer(wx.VERTICAL) 
        self.browser = wx.html2.WebView.New(self) 
        sizer.Add(self.browser, 1, wx.EXPAND, 10) 
        self.SetSizer(sizer) 
        self.SetSize((1000, 800))
        
        #Icon made by turkkub from www.flaticon.com 
        self.SetIcon(wx.Icon(os.path.dirname(os.path.realpath(__file__)) + '/img/worldwide_ico.png', wx.BITMAP_TYPE_PNG))
        
class geeMap:
    """
    Class to add Google Earth Engine Layers to a Folium Map.
    """
    def __init__(self, feature=None, zoom=2):
        """
        Initializes the map object.

        Args:
            feature: Google Earth Engine Feature.
            zoom: The initial zoom level of the map.
        """

        if not feature:
            self.centroid = [0.0,0.0]
        else:
            self.centroid = ee.Geometry.centroid(feature).getInfo()['coordinates'][::-1]

        self.zoom = zoom

        # Open Street Map Base    
        self.map = folium.Map(location=self.centroid, tiles="OpenStreetMap", zoom_start=self.zoom, control_scale=True)

    def addLayer(self, img, viz_params=None, name=''):
        """
        Add Google Earth Engine tile layer in map.

        Args:
            img: Google Earth Engine image.
            viz_params: Vizualization parameters of image.
            name: Layer name to plot on map.
        """
        folium_kwargs={'overlay':True,'name':name}

        img_info = img.getMapId(viz_params)
        mapid = img_info['mapid']
        token = img_info['token']
        folium_kwargs['attr'] = ('Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a> ')
        folium_kwargs['tiles'] = "https://earthengine.googleapis.com/map/%s/{z}/{x}/{y}?token=%s"%(mapid,token)

        layer = folium.TileLayer(**folium_kwargs)
        layer.add_to(self.map) 
   
    def addControls(self):
        self.map.add_child(folium.LayerControl())

    def show(self):
        
        temporaryFile = tempfile.NamedTemporaryFile().name + '.html'

        self.map.save(temporaryFile)
        
        app = wx.App() 
        frame = mapBrowser(None, -1) 
        path = 'file://'+ os.path.abspath(temporaryFile)

        frame.SetTitle('Geepy')
        frame.browser.LoadURL(path) 
        frame.CenterOnScreen()
        frame.Show() 

        app.MainLoop()
 
