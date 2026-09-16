-- Offline ActionMap reload contract for gvd/main.lua (no BeamNG).
-- Run: lua5.1 scripts/test_gvd_bind_reload.lua
-- Proves: core_input_actions.load/loadActions then core_input_bindings.reloadBindings;
-- never call bindings.loadActions; bindReady only after a real reload; pcall failures log;
-- onFileChanged uses (filename, type).

local logs = {}
function log(level, tag, msg) logs[#logs + 1] = tostring(msg) end
function jsonDecode(_) return nil end

local home = os.getenv('HOME') or os.getenv('USERPROFILE')
assert(home and home ~= '', 'HOME/USERPROFILE required')
os.execute('mkdir -p "' .. home .. '/Documents/GVD"')

be = { getPlayerVehicle = function() return nil end }
guihooks = nil
FS = nil

local function check(cond, msg)
  if not cond then error('FAIL: ' .. msg, 2) end
  print('ok   ' .. msg)
end

local function readyIn(msgs)
  for _, m in ipairs(msgs) do
    if tostring(m):find('Alt%+G action ready', 1) then return true end
  end
  return false
end

local function failedIn(msgs, label)
  for _, m in ipairs(msgs) do
    if tostring(m):find(label, 1, true) and tostring(m):find('failed:', 1, true) then
      return true
    end
  end
  return false
end

local ACTION = '/lua/ge/extensions/core/input/actions/gvd.json'

-- A: load then reloadBindings. Title/desc appear on load (premature-ready trap).
do
  logs = {}
  local trace = {}
  local acts = {}
  local prevLog = log
  function log(level, tag, msg)
    prevLog(level, tag, msg)
    if tostring(msg):find('Alt%+G action ready', 1) then trace[#trace + 1] = 'ready' end
  end
  core_input_actions = {
    load = function()
      trace[#trace + 1] = 'load'
      acts.gvd_toggle_engage = {
        title = 'GVD Engage', desc = 'Toggle GVD engage',
        onDown = 'extensions.gvd_main.toggleEngage()',
      }
    end,
    loadActions = function()
      error('loadActions must not run when load exists')
    end,
    getActiveActions = function() return acts end,
  }
  core_input_bindings = {
    loadActions = function()
      error('must never call core_input_bindings.loadActions')
    end,
    onFileChanged = function(path, typ)
      trace[#trace + 1] = 'notify'
      check(typ == 'added', 'onFileChanged type is added')
    end,
    reloadBindings = function() trace[#trace + 1] = 'reload' end,
  }
  extensions = { core_input_actions = core_input_actions, core_input_bindings = core_input_bindings }
  local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
  extensions.gvd_main = M
  M.onExtensionLoaded()
  local loadAt, reloadAt, readyAt
  for i, ev in ipairs(trace) do
    if ev == 'load' and not loadAt then loadAt = i end
    if ev == 'reload' and not reloadAt then reloadAt = i end
    if ev == 'ready' and not readyAt then readyAt = i end
  end
  check(loadAt ~= nil, 'A: core_input_actions.load called')
  check(reloadAt ~= nil, 'A: core_input_bindings.reloadBindings called')
  check(readyAt ~= nil, 'A: Alt+G ready logged')
  check(loadAt < reloadAt, 'A: load before reloadBindings')
  check(reloadAt < readyAt, 'A: ready only after reloadBindings')
  log = prevLog
end

-- B: no reloadBindings (some 0.39 dumps). onFileChanged(filename, 'added') is delayed.
-- title/desc on getActiveActions must not ready on the first tick.
do
  logs = {}
  local notified = 0
  local types = {}
  local acts = {}
  core_input_actions = {
    loadActions = function()
      acts.gvd_toggle_engage = {
        title = 'GVD Engage', desc = 'Toggle GVD engage',
        onDown = 'extensions.gvd_main.toggleEngage()',
      }
    end,
    getActiveActions = function() return acts end,
  }
  core_input_bindings = {
    loadActions = function()
      error('must never call core_input_bindings.loadActions')
    end,
    onFileChanged = function(path, typ)
      notified = notified + 1
      types[#types + 1] = typ
      check(typ == 'added', 'B: onFileChanged arity is (filename, added)')
    end,
  }
  extensions = { core_input_actions = core_input_actions, core_input_bindings = core_input_bindings }
  local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
  extensions.gvd_main = M
  M.onExtensionLoaded()
  check(not readyIn(logs), 'B: bindReady not set before delayed reload completes')
  check(notified >= 1, 'B: onFileChanged scheduled a bindings refresh')
  local notifyAtReady = notified
  for _ = 1, 24 do M.onUpdate(0.05) end
  check(readyIn(logs), 'B: Alt+G ready after waiting past forceRefresh(0.1)')
  check(notified == notifyAtReady, 'B: retries do not reset the forceRefresh(0.1) timer')
end

-- C: pcall failures are logged, not swallowed.
do
  logs = {}
  local acts = {}
  core_input_actions = {
    load = function()
      acts.gvd_toggle_engage = {
        title = 'GVD Engage', desc = 'Toggle GVD engage',
        onDown = 'extensions.gvd_main.toggleEngage()',
      }
    end,
    getActiveActions = function() return acts end,
  }
  core_input_bindings = {
    loadActions = function()
      error('must never call core_input_bindings.loadActions')
    end,
    reloadBindings = function() error('boom') end,
  }
  extensions = { core_input_actions = core_input_actions, core_input_bindings = core_input_bindings }
  local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
  extensions.gvd_main = M
  M.onExtensionLoaded()
  check(failedIn(logs, 'core_input_bindings.reloadBindings'), 'C: reloadBindings pcall failure is logged')
  check(not readyIn(logs), 'C: failed reload does not set bindReady')
end

print('test_gvd_bind_reload: OK')
