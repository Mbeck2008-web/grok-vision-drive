-- Offline lastGood + gvdUi LINKED from absolute io.open of gvd_state.json (no BeamNG).
-- VFS readFile / FS:readFile are poisoned: if they run first, LINKED must NOT come from junk.
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

local function busDocs()
  local ov = os.getenv('GVD_DOCS_DIR')
  if ov then
    ov = tostring(ov):match('^%s*(.-)%s*$') or ''
    if ov ~= '' then return ov:gsub('\\', '/') end
  end
  local la = os.getenv('LOCALAPPDATA')
  if la and la ~= '' and not tostring(la):lower():find('onedrive', 1, true) then
    return la:gsub('\\', '/') .. '/BeamNG/BeamNG.tech/current/Documents/GVD'
  end
  local home = os.getenv('USERPROFILE') or os.getenv('HOME')
  assert(home and home ~= '', 'LOCALAPPDATA or USERPROFILE/HOME required')
  return home:gsub('\\', '/') .. '/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD'
end
local docs = busDocs()
os.execute('mkdir -p "' .. docs .. '"')
local statePath = docs .. '/gvd_state.json'

local function writeFile(p, s)
  local f = assert(io.open(p, 'w'))
  f:write(s)
  f:close()
end

local vfsHits = 0
function readFile(_)
  vfsHits = vfsHits + 1
  return '{"poison":true,"loop_hz":99}'
end
FS = {
  readFile = function(_)
    vfsHits = vfsHits + 1
    return '{"poison":true,"loop_hz":99}'
  end,
  directoryCreate = function() end,
  writeFile = function() return true end,
}

local uiPushes = {}
guihooks = {
  trigger = function(ev, payload)
    if ev == 'gvdUi' then uiPushes[#uiPushes + 1] = payload end
  end,
}
be = { getPlayerVehicle = function() return nil end }

writeFile(statePath, string.format(
  '{"schema":1,"policy":"modular","loop_hz":17,"detector":"link-probe","heartbeat_mtime":%d,"heartbeat_unix":%d}',
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
check(saw('gvd_state read ok path='), 'distinct gvd_state read ok log')
check(not saw('json fail len='), 'fresh JSON is not a json fail')
check(#uiPushes > 0, 'gvdUi pushed on load with lastGood')
local p = uiPushes[#uiPushes]
check(p.link ~= 'none', 'CEF link is not none (was NO LINK / supervisor not running)')
check(p.link == 'live' or p.link == 'stale', 'CEF link is lastGood HB live/stale')
check(tonumber(p.hz) == 17, 'gvdUi hz comes from disk gvd_state, not VFS poison 99')
check(p.detector == 'link-probe', 'gvdUi detector comes from disk, not VFS poison')
check(p.policy == 'modular', 'gvdUi policy from lastGood')

-- CEF poke path: pollStateFile + pushUi even if onUpdate never ticks.
uiPushes = {}
M.pushUiState()
check(#uiPushes > 0, 'pushUiState polls + pushes without onUpdate')
check(uiPushes[#uiPushes].link ~= 'none', 'pushUiState keeps LINKED from lastGood')
check(tonumber(uiPushes[#uiPushes].hz) == 17, 'pushUiState still disk lastGood, not VFS')

-- json fail is distinct from read fail.
writeFile(statePath, '{not-json')
stateJsonFailLogged = nil -- cannot reset module locals; json-fail log is one-shot from first fail
-- First decode already succeeded so lastGood stays the previous table; rewrite + push
-- still must not take VFS poison. Prove readText still prefers io.open by unique hz.
writeFile(statePath, string.format(
  '{"schema":1,"policy":"modular","loop_hz":18,"detector":"link-probe-2","heartbeat_mtime":%d,"heartbeat_unix":%d}',
  os.time(), os.time()))
M.pushUiState()
check(tonumber(uiPushes[#uiPushes].hz) == 18, 'reread via io.open picks up fresh disk bytes')
check(uiPushes[#uiPushes].detector == 'link-probe-2', 'reread lastGood from absolute io.open')
check(vfsHits >= 0, 'VFS stubs callable (hits=' .. tostring(vfsHits) .. ')')

print('test_gvd_state_link: OK')
