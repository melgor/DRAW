-- @Author: blcv
-- @Date:   2015-12-08 14:23:00
-- @Last Modified by:   blcv
-- @Last Modified time: 2015-12-10 19:42:59
package.path = "draw_nngraph/?.lua;" .. package.path
require 'nn'
require 'nngraph'
require 'attention'

local READ = {}

function READ.create(x, x_error_prev, rnn_size, size_image, size_attention, batchSize)
  --read
  h_dec_prev = nn.Identity()()
  createParameters(h_dec_prev, rnn_size, size_image,size_attention)
  ascending = nn.Identity()()
  --gen filters
  filterbank_x = genrFilters(gx, ascending, size_image, size_attention, sigma, gamma, batchSize)
  filterbank_y = genrFilters(gy, ascending, size_image, size_attention, sigma, gamma, batchSize)
  --read data from input
  patch       = applyFilter(x, filterbank_x, filterbank_y)
  patch_error = applyFilter(x_error_prev, filterbank_x, filterbank_y)
  --combine outout and previous error
  read_input = nn.JoinTable(3)({patch, patch_error})
  read_input = nn.Reshape(2 * size_attention * size_attention)(read_input)
  
  return nn.gModule({x, x_error_prev, h_dec_prev,ascending}, {patch, read_input})
end

return READ