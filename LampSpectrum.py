class LampSpectrum:
  def __init__(self, name, data):
    self.name = name
    self.data = data

 def __display__(self,cnv,w,h):
     for pair in self.data:
         cv2.line(cnv, (0, 0), (width, height), white)
         cv2.imshow("Canvas", canvas)
         cv2.waitKey(0)