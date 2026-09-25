-- Offline lastGood + gvdUi from VFS / FS:readFile of relative Documents/GVD/gvd_state.json.
-- Absolute io.open of the Tech/Drive sandbox is the FAIL (Tech GELua cannot io.open it).
-- This is NOT a live BeamNG Apps LINK proof. Live CEF LINK stays UNPROVEN.
-- Run from repo root: lua5.1 scripts/test_gvd_state_link.lua
local logs = {}
function log(level, tag, msg) logs[#logs + 1] = tostring(level) .. ' ' .. tostring(msg) end

local function check(cond, msg)
  if not cond then error('FAIL: ' .. msg, 2) end
  print('ok   ' .. msg)
end

function jsonDecode(s)
  if type(s) ~= 'string' or not s:find('"heartbeat_mtime"', 1, true) then return nil end
  local t = {}
  for k, v in s:gmatch('"([%w_]+)"%s*:%s*([^,}]+)') do
    v = v:gsub('^%s+', ''):gsub('%s+$', '')
    if v == 'true' then t[k] = true
    elseif v == 'false' then t[k] = false
    elseif v == 'null' then t[k] = nil
    elseif v:sub(1, 1) == '"' then t[k] = v:sub(2, -2)
    else t[k] = tonumber(v) end
  end
  if t.heartbeat_mtime == nil then return nil end
  return t
end

local vfsBus = {}
local vfsHits = 0
local function busRel(path)
  local p = tostring(path or ''):gsub('\\', '/')
  local name = p:match('([^/]+)$')
  if name and name:match('^gvd_') then return 'Documents/GVD/' .. name end
  return p
end
local function vfsGet(path)
  vfsHits = vfsHits + 1
  return vfsBus[busRel(path)]
end
function readFile(path)
  return vfsGet(path)
end
function writeFile(path, data)
  vfsBus[busRel(path)] = data
  return true
end
FS = {
  readFile = function(_, path)
    return vfsGet(path)
  end,
  directoryCreate = function() end,
  getUserPath = function()
    return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current'
  end,
}

-- Absolute disk poison: if readText still io.opens the sandbox, lastGood would take this.
local poisonDocs
do
  local la = os.getenv('LOCALAPPDATA')
  local home = os.getenv('USERPROFILE') or os.getenv('HOME')
  local root = la or ((home and home ~= '') and (tostring(home):gsub('\\', '/') .. '/AppData/Local') or nil)
  if root then
    poisonDocs = tostring(root):gsub('\\', '/') .. '/BeamNG/BeamNG.tech/current/Documents/GVD'
    os.execute('mkdir -p "' .. poisonDocs .. '"')
    local pf = assert(io.open(poisonDocs .. '/gvd_state.json', 'w'))
    pf:write('{"schema":1,"policy":"poison-abs","loop_hz":99,"detector":"abs-io-open","heartbeat_mtime":1,"heartbeat_unix":1}')
    pf:close()
  end
end

local busReadOpens = 0
local realOpen = io.open
io.open = function(path, mode)
  local p = tostring(path or '')
  local m = tostring(mode or 'r')
  if m:sub(1, 1) == 'r' and (p:find('gvd_state', 1, true) or p:find('Documents/GVD', 1, true)
      or p:find('Documents\\GVD', 1, true)) then
    busReadOpens = busReadOpens + 1
    return nil
  end
  return realOpen(path, mode)
end

local function writeBus(name, s)
  vfsBus['Documents/GVD/' .. name] = s
end

local uiPushes = {}
guihooks = {
  trigger = function(ev, payload)
    if ev == 'gvdUi' then uiPushes[#uiPushes + 1] = payload end
  end,
}
be = { getPlayerVehicle = function() return nil end }

writeBus('gvd_state.json', string.format(
  '{"schema":1,"policy":"modular","loop_hz":17,"detector":"link-probe","heartbeat_mtime":%d,"heartbeat_unix":%d,"python_bus":"C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD","lua_bus":"C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD","product":"drive","bus_link":"ok","cmd_seq":4}',
  os.time(), os.time()))

local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
extensions = { gvd_main = M }
M.onExtensionLoaded()

local function saw(needle)
  for _, m in ipairs(logs) do
    if tostring(m):find(needle, 1, true) then return true end
  end
  return false
end

check(saw('docs dir='), 'gvdDocsDir logged once on load')
check(saw('docs dir=Documents/GVD'), 'docs dir is relative Documents/GVD')
check(saw('gvd_state read ok path='), 'distinct gvd_state read ok log')
check(saw('gvd_state read ok path=Documents/GVD/gvd_state.json'),
  'read ok path is relative Documents/GVD/gvd_state.json')
check(not saw('json fail len='), 'fresh JSON is not a json fail')
check(#uiPushes > 0, 'gvdUi pushed on load with lastGood')
local p = uiPushes[#uiPushes]
check(p.link ~= 'none', 'CEF link is not none (was NO LINK / supervisor not running)')
check(p.link == 'live' or p.link == 'stale', 'CEF link is lastGood HB live/stale')
check(p.link ~= 'mismatch', 'matching python_bus/lua_bus is not MISMATCH')
check(p.luaBus == 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
  'gvdUi luaBus is resolved Documents/GVD')
check(p.pythonBus == 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
  'gvdUi pythonBus comes from gvd_state.json')
check(p.product == 'drive', 'gvdUi product=drive from userfolder')
check(tonumber(p.hz) == 17, 'gvdUi hz comes from VFS Documents/GVD, not absolute io.open poison 99')
check(p.detector == 'link-probe', 'gvdUi detector comes from VFS, not abs io.open')
check(p.policy == 'modular', 'gvdUi policy from lastGood')
check(busReadOpens == 0, 'no io.open for bus reads (opens=' .. tostring(busReadOpens) .. ')')

uiPushes = {}
M.pushUiState()
check(#uiPushes > 0, 'pushUiState polls + pushes without onUpdate')
check(uiPushes[#uiPushes].link ~= 'none', 'pushUiState keeps LINKED from lastGood')
check(tonumber(uiPushes[#uiPushes].hz) == 17, 'pushUiState still VFS lastGood, not abs io.open')

-- reread via VFS picks up fresh bytes. Absolute disk poison must not win.
writeBus('gvd_state.json', string.format(
  '{"schema":1,"policy":"modular","loop_hz":18,"detector":"link-probe-2","heartbeat_mtime":%d,"heartbeat_unix":%d,"python_bus":"C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD","product":"drive","cmd_seq":5}',
  os.time(), os.time()))
M.pushUiState()
check(tonumber(uiPushes[#uiPushes].hz) == 18, 'reread via VFS picks up fresh Documents/GVD bytes')
check(uiPushes[#uiPushes].detector == 'link-probe-2', 'reread lastGood from VFS, not absolute io.open')
check(busReadOpens == 0, 'reread still does not io.open the bus')
check(vfsHits > 0, 'VFS stubs served the bus (hits=' .. tostring(vfsHits) .. ')')

-- Flip of the old absolute-io.open success: VFS empty, only abs disk exists → read fail, keep lastGood.
vfsBus['Documents/GVD/gvd_state.json'] = nil
local hzBefore = tonumber(uiPushes[#uiPushes].hz)
M.pushUiState()
check(tonumber(uiPushes[#uiPushes].hz) == hzBefore,
  'empty VFS does not fall back to absolute io.open (lastGood stays 18)')
check(busReadOpens == 0, 'empty VFS still does not io.open abs Tech sandbox')

-- Drive Lua + Tech python_bus is MISMATCH (do not guess).
writeBus('gvd_state.json', string.format(
  '{"schema":1,"policy":"modular","loop_hz":19,"detector":"link-probe-3","heartbeat_mtime":%d,"heartbeat_unix":%d,"python_bus":"C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD","product":"tech","cmd_seq":6}',
  os.time(), os.time()))
M.pushUiState()
check(uiPushes[#uiPushes].link == 'mismatch', 'Tech python_bus vs Drive lua_bus is link=MISMATCH')
M.onUpdate(1.1)
check(saw('[GVD][LUA]'), '1 Hz LUA bus line is printed')
check(saw('bus='), 'LUA line prints bus=')
check(saw('link=MISMATCH'), '1 Hz identity prints link=MISMATCH')

print('test_gvd_state_link: OK')
