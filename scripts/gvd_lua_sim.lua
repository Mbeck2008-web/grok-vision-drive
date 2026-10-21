-- Step server for beamng_mod/lua/ge/extensions/gvd/main.lua on a real bus folder (no BeamNG).
-- Not a test itself (the suite only runs test_*). scripts/test_frozen_supervisor_deadman.py
-- drives it: Python writes gvd_cmd.json / gvd_state.json on disk, this reads them through
-- the same GE stubs as test_m6_lua_cmd.lua and reports what the mod does to the vehicle.
--
--   GVD_SIM_USER=<userfolder> lua5.1 scripts/gvd_lua_sim.lua      (cwd = repo root)
--
-- stdin, one command per line:  step <dt_s> [<wall_s>] | engage | quit
-- With <wall_s> (sim wall time at the end of that frame, Unix seconds) the mod's wall clock
-- follows the sim (whole seconds, like os.time) instead of the host's os.time.
-- stdout per command: "ev <input> <value>" for each new input.event, then
-- "st <engaged 0|1>", then "ok".

local USER = assert(os.getenv('GVD_SIM_USER'), 'GVD_SIM_USER is the BeamNG userfolder')
USER = USER:gsub('\\', '/'):gsub('/$', '')
local BUS = USER .. '/Documents/GVD'

function log(level, tag, msg) io.stderr:write(tostring(msg), '\n') end

-- Same flat decoder as test_m6_lua_cmd.lua (BeamNG provides jsonDecode in-game).
local function decodeScalars(s)
  local t = {}
  for k, v in s:gmatch('"([%w_]+)"%s*:%s*([^,}]+)') do
    v = v:gsub('^%s+', ''):gsub('%s+$', '')
    if v == 'true' then t[k] = true
    elseif v == 'false' then t[k] = false
    elseif v == 'null' then t[k] = nil
    elseif v:sub(1, 1) == '"' then t[k] = v:sub(2, -2)
    else t[k] = tonumber(v) end
  end
  return t
end
function jsonDecode(s)
  local t = {}
  local rest = s:gsub('"([%w_]+)"%s*:%s*(%b{})', function(k, obj)
    t[k] = decodeScalars(obj)
    return ''
  end)
  for k, v in pairs(decodeScalars(rest)) do t[k] = v end
  return t
end

-- VFS Documents/GVD/<gvd_*> -> <userfolder>/Documents/GVD/<gvd_*> on disk.
local function diskPath(path)
  local p = tostring(path or ''):gsub('\\', '/')
  local name = p:match('([^/]+)$')
  if name and name:match('^gvd_') then return BUS .. '/' .. name end
  return nil
end
local function readDisk(path)
  local p = diskPath(path)
  if not p then return nil end
  local f = io.open(p, 'rb')
  if not f then return nil end
  local s = f:read('*a')
  f:close()
  if s == '' then return nil end
  return s
end
local function writeDisk(path, data)
  local p = diskPath(path)
  if not p then return false end
  local f = io.open(p, 'wb')
  if not f then return false end
  f:write(data)
  f:close()
  return true
end
function readFile(path) return readDisk(path) end
function writeFile(path, data) return writeDisk(path, data) end
FS = {
  readFile = function(_, path) return readDisk(path) end,
  writeFile = function(_, path, data) return writeDisk(path, data) end,
  directoryCreate = function() end,
  getUserPath = function() return USER end,
}

local events = {}
local geQueue = {}
local electricsValues = { wheelspeed = 0, steering_input = 0, throttle_input = 0, brake_input = 0 }
local veEnv = {
  input = {
    -- The car follows what it is given: the electrics echo is the applied input, so the
    -- override residual sees no player.
    event = function(itype, ivalue)
      events[#events + 1] = { itype, ivalue }
      local key = ({ steering = 'steering_input', throttle = 'throttle_input', brake = 'brake_input' })[itype]
      if key then electricsValues[key] = ivalue end
    end,
    setAllowedInputSource = function() end,
    lastInputs = {},
  },
  electrics = { values = electricsValues },
  drivetrain = { setShifterMode = function() end },
  sensors = { gx2 = 0, gy2 = 0, gz2 = 0 },
  obj = {
    queueGameEngineLua = function(_, s) geQueue[#geQueue + 1] = s end,
    getYawAngularVelocity = function() return 0.0 end,
  },
  pcall = pcall, tonumber = tonumber, tostring = tostring, string = string, math = math,
  pairs = pairs, type = type,
}
local Veh = {}
Veh.__index = Veh
function Veh:queueLuaCommand(code)
  local fn, err = loadstring(code, 'vehicle-lua')
  assert(fn, 'vehicle snippet does not compile: ' .. tostring(err))
  setfenv(fn, veEnv)
  fn()
end
function Veh:getPosition() return { x = 0, y = 0, z = 0 } end
function Veh:getDirectionVector() return { x = 0, y = 1, z = 0 } end
function Veh:getID() return self.id end
local veh = setmetatable({ id = 101 }, Veh)
be = { getPlayerVehicle = function() return veh end }

local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
extensions = { gvd_main = M }
local actions = {}
core_input_actions = {
  actionsCache = {}, normalActionsCache = {},
  load = function()
    actions.gvd_toggle_engage = { title = 'GVD Engage', desc = 'Toggle GVD engage',
      onDown = 'extensions.gvd_main.toggleEngage()' }
  end,
  onFileChanged = function() end,
  getActiveActions = function() return actions end,
}
core_input_bindings = { onFileChanged = function() end, reloadBindings = function() end }
extensions.core_input_actions = core_input_actions
extensions.core_input_bindings = core_input_bindings

local function drainGE()
  local q = geQueue
  geQueue = {}
  for _, s in ipairs(q) do
    local fn = loadstring(s, 'ge-lua')
    if fn then fn() end
  end
end

local function reply()
  for _, ev in ipairs(events) do
    io.stdout:write(string.format('ev %s %.4f\n', tostring(ev[1]), tonumber(ev[2]) or 0))
  end
  events = {}
  io.stdout:write(string.format('st %d\n', M.isEngaged() and 1 or 0))
  io.stdout:write('ok\n')
  io.stdout:flush()
end

local simWall = nil
local function simWallClock() return math.floor(simWall) end

M.onExtensionLoaded()
reply()
for line in io.stdin:lines() do
  local cmd, arg, arg2 = line:match('^(%S+)%s*(%S*)%s*(%S*)')
  if cmd == 'quit' then break end
  if cmd == 'engage' then
    M.toggleEngage()
  elseif cmd == 'step' then
    local wall = tonumber(arg2)
    if wall then
      -- A mod without the seam keeps the host os.time (the slow-runner check then fails).
      if simWall == nil and M.setWallClockForTest then M.setWallClockForTest(simWallClock) end
      simWall = wall
    end
    M.onUpdate(tonumber(arg) or 0)
    drainGE()
  end
  reply()
end
