-- Offline harness for gvdDocsDir Tech sandbox (no BeamNG).
-- Extracts the live helpers from main.lua and asserts:
--   GVD_DOCS_DIR → override
--   LOCALAPPDATA → …/BeamNG/BeamNG.tech/current/Documents/GVD
--   FS AppData/Local or BeamNG.tech/current → same Tech sandbox
--   never USERPROFILE/Documents/GVD, never a bare gvd_*.json under current\
-- Run from repo root: lua5.1 scripts/test_gvd_docs_dir.lua

local function check(cond, msg)
  if not cond then error('FAIL: ' .. msg, 2) end
  print('ok   ' .. msg)
end

local f = assert(io.open('beamng_mod/lua/ge/extensions/gvd/main.lua', 'r'))
local src = f:read('*a')
f:close()

local i = src:find('local gvdDocsResolved', 1, true)
local j = src:find('\nlocal function userEngagePath', i, true)
check(i and j, 'main.lua exports gvdDocsDir helpers before userEngagePath')
local body = src:sub(i, j - 1)
check(body:find('_localAppFromPath', 1, true), 'extracted _localAppFromPath')
check(body:find('_techCurrentFromPath', 1, true), 'extracted _techCurrentFromPath')
check(body:find('GVD_DOCS_DIR', 1, true), 'GVD_DOCS_DIR override is in helpers')
check(body:find('LOCALAPPDATA', 1, true), 'LOCALAPPDATA Tech sandbox is in helpers')
check(body:find('BeamNG/BeamNG.tech/current/Documents/GVD', 1, true), 'Tech tail is in helpers')
check(not body:find('USERPROFILE') or body:find('AppData/Local', 1, true),
  'USERPROFILE only synthesizes AppData/Local, not Documents/GVD')

local loadfn = loadstring or load
local function loadHelpers(envVars, fs)
  local sandbox = {
    os = {
      getenv = function(k)
        if envVars == nil then return nil end
        return envVars[k]
      end,
    },
    FS = fs,
    log = function() end,
    print = function() end,
    pcall = pcall,
    tostring = tostring,
    ipairs = ipairs,
    pairs = pairs,
    type = type,
    string = string,
    table = table,
    math = math,
    assert = assert,
  }
  local chunk = body .. '\nreturn { localApp = _localAppFromPath, techCurrent = _techCurrentFromPath, docsDir = gvdDocsDir, file = gvdFile }\n'
  local fn, err = loadfn(chunk)
  check(fn, 'helpers compile: ' .. tostring(err))
  if setfenv then setfenv(fn, sandbox) else
    -- Lua 5.2+: inject env via _ENV in a wrapper. 5.1/luajit is the target.
    error('FAIL: setfenv required (lua5.1 / luajit)')
  end
  return fn()
end

local H = loadHelpers({}, nil)
local SPEC = 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD'

-- ── localApp / techCurrent ───────────────────────────────────────────────────
check(H.localApp('C:/Users/Name/AppData/Local/BeamNG.drive/0.36') == 'C:/Users/Name/AppData/Local',
  'AppData/Local prefix from Drive userfolder')
check(H.localApp('C:\\Users\\Name\\AppData\\Local\\BeamNG\\BeamNG.tech\\current') == 'C:/Users/Name/AppData/Local',
  'AppData/Local prefix with backslashes')
check(H.localApp('C:/Users/Name/Documents/BeamNG.drive/current') == nil,
  'Documents userfolder does not invent LOCALAPPDATA')
check(H.localApp(nil) == nil, 'nil path → nil')
check(H.localApp('') == nil, 'empty path → nil')
check(H.techCurrent('C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current')
    == 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current',
  'nested BeamNG.tech/current is the Tech userfolder')
check(H.techCurrent('C:/Users/Name/AppData/Local/BeamNG.tech/current') == nil,
  'legacy BeamNG.tech (no BeamNG parent) is not the spec current')
check(H.techCurrent('C:/Users/Name/OneDrive/AppData/Local/BeamNG/BeamNG.tech/current') == nil,
  'OneDrive tech current is rejected')
check(H.localApp('C:/Users/Name/OneDrive/AppData/Local/foo') == nil,
  'OneDrive AppData/Local is rejected')

-- ── resolve ──────────────────────────────────────────────────────────────────
local function resolve(envVars, fs)
  return loadHelpers(envVars, fs).docsDir()
end

check(resolve({ GVD_DOCS_DIR = 'D:\\custom\\GVD', LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local' }, nil)
    == 'D:/custom/GVD',
  'GVD_DOCS_DIR wins over LOCALAPPDATA')

check(resolve({ LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local', USERPROFILE = 'C:\\Users\\Name' }, {
      getUserPath = function() return 'C:/Users/Name/Documents/BeamNG.drive/0.36/current' end,
      directoryCreate = function() end,
    }) == SPEC,
  'LOCALAPPDATA Tech sandbox wins over USERPROFILE and Documents FS path')

check(resolve({ USERPROFILE = 'C:\\Users\\Name' }, nil) == SPEC,
  'USERPROFILE synthesizes AppData/Local Tech sandbox (not Documents/GVD)')

check(resolve({ HOME = '/home/me' }, nil)
    == '/home/me/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD',
  'HOME synthesizes AppData/Local Tech sandbox when LOCALAPPDATA empty')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/OneDrive', USERPROFILE = 'C:/Users/Name' }, nil) == SPEC,
  'OneDrive LOCALAPPDATA is rejected; USERPROFILE AppData/Local is used')

check(resolve({}, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current' end,
      directoryCreate = function() end,
    }) == SPEC,
  'FS Tech current userfolder → Tech Documents/GVD')

check(resolve({}, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG.drive/0.36' end,
      directoryCreate = function() end,
    }) == SPEC,
  'FS Drive AppData userfolder reconstructs LOCALAPPDATA Tech sandbox')

check(resolve({}, {
      virtual2Native = function(_, vp)
        if vp == 'settings' then return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/settings' end
      end,
      directoryCreate = function() end,
    }) == SPEC,
  'FS:virtual2Native Tech settings path → Tech Documents/GVD')

check(resolve({}, {
      getUserPath = function() return 'C:/Users/Name/Documents/BeamNG.drive/0.36/current' end,
      directoryCreate = function() end,
    }) == 'Documents/GVD',
  'FS Documents userfolder does NOT become USERPROFILE/Documents/GVD')

-- last resort is Documents/GVD, never a bare filename, never userfolder current\
local last = resolve({}, { directoryCreate = function() end })
check(last == 'Documents/GVD', 'last resort is Documents/GVD (not CWD, not current\\)')
check(last ~= 'current' and not last:match('gvd_.*%.json$'),
  'last resort is not a bare gvd_*.json')

-- gvdFile never returns a bare filename
local files = loadHelpers({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, { directoryCreate = function() end })
for _, name in ipairs({'gvd_state.json', 'gvd_ego.json', 'gvd_engage.json', 'gvd_cmd.json'}) do
  local p = files.file(name)
  check(p == SPEC .. '/' .. name, 'gvdFile(' .. name .. ') under Tech current/Documents/GVD')
  check(not p:match('^gvd_'), name .. ' is not a bare filename')
  check(not p:match('/current/' .. name .. '$'), name .. ' is not under userfolder current\\')
end

-- mkdir is attempted on the resolved docs dir
local mkdirDir
local made = loadHelpers({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, {
  directoryCreate = function(_, dir) mkdirDir = dir end,
})
made.docsDir()
check(mkdirDir == SPEC, 'mkdir of resolved Tech Documents/GVD')

check(body:find("docs dir=", 1, true) and body:find('gvdDocsLogged', 1, true),
  'one-shot log of resolved docs dir')

print('test_gvd_docs_dir: OK')
