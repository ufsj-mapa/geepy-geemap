import folium
import ee
ee.Initialize()

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
        
    def show(self):
        """
        Show map in jupyter-notebook.
        """
        return self.map
   
    def showWindow(self):
        """
        Show map in a separate window.
        """
        return self.map._repr_html_() 
        
    def addControls(self):
        """
        Add controls to map.
        """
        return self.map.add_child(folium.LayerControl())
