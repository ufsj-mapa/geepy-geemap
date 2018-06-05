# geepy Map Browser
import wx 
import wx.html2 
import os

#import ee.Initialize()

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
        self.SetIcon(wx.Icon('worldwide.png', wx.BITMAP_TYPE_PNG)) 

if __name__ == '__main__': 
    app = wx.App() 
    frame = mapBrowser(None, -1) 
    path = 'file://'+ os.path.abspath('index.html')

    frame.browser.LoadURL(path) 
    frame.CenterOnScreen()
    frame.Show() 
  
    app.MainLoop() 
