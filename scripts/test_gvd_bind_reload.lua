-- Offline ActionMap reload contract for gvd/main.lua (no BeamNG).
-- Run: lua5.1 scripts/test_gvd_bind_reload.lua
-- Proves: always clear actionsCache then load/loadActions then reloadBindings;
-- never call bindings.loadActions; bindReady after reload AND dump-shape listing
-- of gvd_toggle_engage (title/desc alone is not enough); keep retrying if unlisted;
-- fail log only if onFileChanged ran and reload still failed; pcall failures log;
-- onFileChanged uses (filename, type); throwing onFileChanged is not success;
-- 0.36/0.39 dump shapes; a path with neither load nor loadActions (onFileChanged + defer).

local logs = {}
function log(level, tag, msg) logs[#logs + 1] = tostring(msg) end
function jsonDecode(_) return nil end

local function busDocs()
  local ov = os.getenv('GVD_DOCS_DIR')
  if ov then
    ov = tostring(ov):match('^%s*(.-)%s*$') or ''
    if ov ~= '' then return ov:gsub('\\', '/') end
  end
  local product = 'drive'
  local gp = os.getenv('GVD_PRODUCT')
  if gp and tostring(gp) ~= '' then
    gp = tostring(gp):lower():match('^%s*(.-)%s*$') or ''
    if gp == 'tech' or gp == 'beamng.tech' or gp == 'beamngtech' then product = 'tech' end
  end
  local b = os.getenv('GVD_BEAMNG')
  if b then
    b = tostring(b):lower():match('^%s*(.-)%s*$') or ''
    if b == '1' or b == 'true' or b == 'yes' then product = 'tech' end
  end
  local be = os.getenv('GVD_BACKEND')
  if be then
    be = tostring(be):lower():match('^%s*(.-)%s*$') or ''
    if be == 'beamngpy' or be == 'tech' then product = 'tech' end
  end
  local function fromLa(localApp)
    if not localApp or localApp == '' then return nil end
    local la = tostring(localApp):gsub('\\', '/')
    if la == '' or la:lower():find('onedrive', 1, true) then return nil end
    if product == 'tech' then
      return la .. '/BeamNG/BeamNG.tech/current/Documents/GVD'
    end
    return la .. '/BeamNG/BeamNG.drive/current/Documents/GVD'
  end
  local d = fromLa(os.getenv('LOCALAPPDATA'))
  if d then return d end
  local home = os.getenv('HOME') or os.getenv('USERPROFILE')
  assert(home and home ~= '', 'HOME/USERPROFILE required')
  return fromLa(home:gsub('\\', '/') .. '/AppData/Local')
end
os.execute('mkdir -p "' .. busDocs() .. '"')

be = { getPlayerVehicle = function() return nil end }
guihooks = nil
FS = nil
function writeFile() return true end

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

-- A: cache clear then load then reloadBindings. Title/desc on load is not the ready signal.
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
    actionsCache = { [true] = { stale = true } },
    normalActionsCache = { [true] = { stale = true } },
    onFileChanged = function(path, typ)
      trace[#trace + 1] = 'cache'
      check(typ == 'added', 'A: actions.onFileChanged type is added')
    end,
    load = function()
      trace[#trace + 1] = 'load'
      check(core_input_actions.actionsCache[true] == nil, 'A: actionsCache[true] cleared before load')
      check(core_input_actions.normalActionsCache[true] == nil, 'A: normalActionsCache[true] cleared before load')
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
  local cacheAt, loadAt, reloadAt, readyAt
  for i, ev in ipairs(trace) do
    if ev == 'cache' and not cacheAt then cacheAt = i end
    if ev == 'load' and not loadAt then loadAt = i end
    if ev == 'reload' and not reloadAt then reloadAt = i end
    if ev == 'ready' and not readyAt then readyAt = i end
  end
  check(cacheAt ~= nil, 'A: actionsCache drop via onFileChanged')
  check(loadAt ~= nil, 'A: core_input_actions.load called')
  check(reloadAt ~= nil, 'A: core_input_bindings.reloadBindings called')
  check(readyAt ~= nil, 'A: Alt+G ready logged')
  check(cacheAt < loadAt, 'A: cache clear before load')
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
  for _ = 1, 48 do M.onUpdate(0.05) end
  check(not readyIn(logs), 'C: ticks after failed reload still do not set bindReady')
end

-- D: first non-throwing reloadBindings is not ready until dump-shape walk lists the action.
do
  logs = {}
  local listed = false
  local def = {
    title = 'GVD Engage', desc = 'Toggle GVD engage',
    onDown = 'extensions.gvd_main.toggleEngage()',
  }
  core_input_actions = {
    load = function() end,
    getActiveActions = function()
      if listed then return { [true] = { gvd_toggle_engage = def } } end
      return {}
    end,
  }
  core_input_bindings = {
    loadActions = function()
      error('must never call core_input_bindings.loadActions')
    end,
    reloadBindings = function() end,
  }
  extensions = { core_input_actions = core_input_actions, core_input_bindings = core_input_bindings }
  local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
  extensions.gvd_main = M
  M.onExtensionLoaded()
  check(not readyIn(logs), 'D: reload without listed action does not set bindReady')
  listed = true
  for _ = 1, 48 do M.onUpdate(0.05) end
  check(readyIn(logs), 'D: bindReady after dump-shape walk lists gvd_toggle_engage')
end

-- E: 0.39-style boolean-keyed action dump. listed sees it; ready is still the reload.
do
  logs = {}
  local def = {
    title = 'GVD Engage', desc = 'Toggle GVD engage',
    onDown = 'extensions.gvd_main.toggleEngage()',
  }
  core_input_actions = {
    loadActions = function() end,
    getActiveActions = function()
      return { [true] = { gvd_toggle_engage = def } }
    end,
  }
  core_input_bindings = {
    loadActions = function()
      error('must never call core_input_bindings.loadActions')
    end,
    reloadBindings = function() end,
  }
  extensions = { core_input_actions = core_input_actions, core_input_bindings = core_input_bindings }
  local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
  extensions.gvd_main = M
  M.onExtensionLoaded()
  check(readyIn(logs), 'E: 0.39 actionsCache[true] dump shape still reloads')
end

-- F: no onFileChanged and failed reload -- do not throw the dead-bind error.
do
  logs = {}
  core_input_actions = {
    load = function() end,
    getActiveActions = function() return {} end,
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
  for _ = 1, 48 do M.onUpdate(0.05) end
  check(not readyIn(logs), 'F: still not ready')
  local dead = false
  for _, m in ipairs(logs) do
    if tostring(m):find('still missing after retries', 1, true) then dead = true end
  end
  check(not dead, 'F: fail log only if onFileChanged actually ran')
end

-- G: neither load nor loadActions. Only onFileChanged + defer. Test B still has loadActions.
do
  logs = {}
  local notified = 0
  local def = {
    title = 'GVD Engage', desc = 'Toggle GVD engage',
    onDown = 'extensions.gvd_main.toggleEngage()',
  }
  core_input_actions = {
    getActiveActions = function()
      return { [true] = { gvd_toggle_engage = def } }
    end,
  }
  core_input_bindings = {
    loadActions = function()
      error('must never call core_input_bindings.loadActions')
    end,
    onFileChanged = function(path, typ)
      notified = notified + 1
      check(typ == 'added', 'G: onFileChanged arity is (filename, added)')
    end,
  }
  extensions = { core_input_actions = core_input_actions, core_input_bindings = core_input_bindings }
  local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
  extensions.gvd_main = M
  M.onExtensionLoaded()
  check(not readyIn(logs), 'G: bindReady not set before delayed reload completes')
  check(core_input_actions.load == nil and core_input_actions.loadActions == nil,
    'G: fixture has neither load nor loadActions')
  check(notified >= 1, 'G: onFileChanged scheduled a bindings refresh')
  local notifyAtReady = notified
  for _ = 1, 24 do M.onUpdate(0.05) end
  check(readyIn(logs), 'G: Alt+G ready after onFileChanged + defer (no load/loadActions)')
  check(notified == notifyAtReady, 'G: retries do not reset the forceRefresh(0.1) timer')
end

-- H: onFileChanged export exists but both pcalls throw; do not markBindReady.
do
  logs = {}
  local def = {
    title = 'GVD Engage', desc = 'Toggle GVD engage',
    onDown = 'extensions.gvd_main.toggleEngage()',
  }
  core_input_actions = {
    getActiveActions = function()
      return { gvd_toggle_engage = def }
    end,
  }
  core_input_bindings = {
    loadActions = function()
      error('must never call core_input_bindings.loadActions')
    end,
    onFileChanged = function() error('boom') end,
  }
  extensions = { core_input_actions = core_input_actions, core_input_bindings = core_input_bindings }
  local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
  extensions.gvd_main = M
  M.onExtensionLoaded()
  check(failedIn(logs, 'core_input_bindings.onFileChanged'), 'H: onFileChanged pcall failure is logged')
  check(not readyIn(logs), 'H: thrown onFileChanged does not set bindReady')
  for _ = 1, 48 do M.onUpdate(0.05) end
  check(not readyIn(logs), 'H: ticks after thrown onFileChanged still do not set bindReady')
end

print('test_gvd_bind_reload: OK')
