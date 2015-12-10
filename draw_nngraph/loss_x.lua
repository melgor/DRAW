-- @Author: blcv
-- @Date:   2015-12-10 08:43:35
-- @Last Modified by:   blcv
-- @Last Modified time: 2015-12-10 09:37:44
require 'nn'
require 'nngraph'

LOSS_X = {}

function LOSS_X.create(x, next_canvas)
  mu = nn.Sigmoid()(next_canvas)

  neg_mu = nn.MulConstant(-1)(mu)
  d = nn.CAddTable()({x, neg_mu})
  d2 = nn.Power(2)(d)
  loss_x = nn.Sum(3)(d2)
  loss_x = nn.Sum(2)(loss_x)


  x_prediction = nn.Reshape(28, 28)(mu)
  x_error = nn.Reshape(28, 28)(d)

  return nn.gModule({next_canvas,x}, {loss_x,x_prediction,x_error})
end
return LOSS_X