# geepy Map Browser

import wx 
import wx.html2 

class mapBrowser(wx.Frame): 
  def __init__(self, *args, **kwds): 
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
  frame.browser.LoadURL("http://www.google.com") 
  frame.CenterOnScreen()
  frame.Show() 
  app.MainLoop() 
