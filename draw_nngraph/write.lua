-- @Author: blcv
-- @Date:   2015-12-08 14:28:06
-- @Last Modified by:   blcv
-- @Last Modified time: 2015-12-10 10:32:23
package.path = "draw_nngraph/?.lua;" .. package.path
require 'nn'
require 'nngraph'
require 'attention'

local WRITE = {}

function WRITE.create(next_h, prev_canvas, rnn_size, size_image, size_attention, batchSize)
   -- get parameter for attention
  createParameters(next_h, rnn_size, size_image, size_attention)
  -- create filter bank based on parameters
  ascending = nn.Identity()()
  filterbank_x = genrFilters(gx, ascending, size_image, size_attention, sigma, gamma, batchSize)
  filterbank_y = genrFilters(gy, ascending, size_image, size_attention, sigma, gamma, batchSize)
  --get prediction
  next_w = nn.Linear(rnn_size, size_attention * size_attention)(next_h)
  next_w = nn.Reshape(size_attention, size_attention)(next_w)
  --get only fragment choosen by filter
  write_layer = nn.MM(true, false)({filterbank_y, next_w})
  write_layer = nn.MM()({write_layer, filterbank_x})
  next_canvas = nn.CAddTable()({prev_canvas, write_layer})
  return  nn.gModule({next_h,ascending, prev_canvas}, {next_canvas})

end
return WRITE