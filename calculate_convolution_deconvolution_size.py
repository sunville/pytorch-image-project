def conv2d_output_size(h,k,s,p):
  dilation=1
  return (h+2*p-dilation*(k-1)-1)/s+1
  
def convtrans2d_output_size(h,k,s,p):
  dilation=1
  out_p = 0
  return (h-1)*s-2*p+dilation*(k-1)+out_p+1
