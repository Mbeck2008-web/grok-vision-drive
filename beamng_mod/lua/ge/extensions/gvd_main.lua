-- Grok Vision Drive — game-side extension (no process inject; BeamNG loads this itself)
local M = {}

local engaged = false

local function onInit()
end

local function logLoaded()
  log('I', 'GVD', '[GVD] loaded. Alt+A engage.')
  print('[GVD] loaded. Alt+A engage.')
end

function M.onExtensionLoaded()
  logLoaded()
end

function M.onExtensionUnloaded()
  engaged = false
  log('I', 'GVD', '[GVD] unloaded.')
end

-- Engage/disengage stub — Python supervisor owns actuators later (M2+)
function M.toggleEngage()
  engaged = not engaged
  local state = engaged and 'ENGAGED' or 'DISENGAGED'
  log('I', 'GVD', '[GVD] ' .. state .. ' (stub).')
  print('[GVD] ' .. state .. ' (stub).')
end

M.onInit = onInit

return M
