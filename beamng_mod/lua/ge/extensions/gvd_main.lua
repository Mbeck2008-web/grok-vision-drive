-- Grok Vision Drive — game-side extension (no process inject; BeamNG loads this itself)
local M = {}

local function logLoaded()
  log('I', 'GVD', '[GVD] loaded. Alt+A engage.')
  print('[GVD] loaded. Alt+A engage.')
end

function M.onExtensionLoaded()
  logLoaded()
end

function M.onExtensionUnloaded()
  log('I', 'GVD', '[GVD] unloaded.')
end

-- Engage/disengage stub — Python supervisor owns actuators later (M2+)
function M.toggleEngage()
  log('I', 'GVD', '[GVD] engage toggled (stub).')
  print('[GVD] engage toggled (stub).')
end

M.onInit = nop

return M
